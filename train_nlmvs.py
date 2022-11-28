import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.dataset import PreProcessedDataset#MVSRDataset
from core.mvs_models import MVSfSNet
from core.sfs_utils import *
from core.mvs_utils import get_depth_values, depth_to_normal, sample_from_depth_prob_volume, sample_from_normal_volume, construct_reflectance_feature_volumes, construct_image_feature_volumes, sample_from_src_views
#from core.sfs_criterion import NormalLogLikelihoodLoss
from core.mvs_criterion import DepthLogLikelihoodLoss, DepthNormalConsistencyLoss

import numpy as np

import os
import glob
from tqdm import tqdm
import argparse

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 4
numdepth = 192

parser = argparse.ArgumentParser()
parser.add_argument('--wo-sfs', action='store_true')
parser.add_argument('--dataset-dir', type=str, default=os.environ['HOME']+'/data/tmp/mvsfs')
args = parser.parse_args()

weight_dir = './weights/nlmvsnet'
if args.wo_sfs:
    weight_dir = './weights/nlmvsnet-wo-sfs'

dataset = PreProcessedDataset(args.dataset_dir)

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

list_split = np.arange(len(dataset))
train_subset_indices =  list_split[:int(0.8*len(list_split))]
train_dataset = Subset(dataset, train_subset_indices)
trainloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

val_subset_indices =  list_split[int(0.8*len(list_split)):int(0.9*len(list_split))]
val_dataset = Subset(dataset, val_subset_indices)
valloader = DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

# load model
mvsfsnet = MVSfSNet(wo_sfs=args.wo_sfs)
if not args.wo_sfs:
    val_loss_min = 1e12
    checkpoint_path = None
    for f in sorted(glob.glob('./weights/sfsnet/*.ckpt')):
        val_loss = torch.load(f)['val_loss']
        if val_loss < val_loss_min:
            checkpoint_path = f
            val_loss_min = val_loss
    checkpoint = torch.load(checkpoint_path)
    mvsfsnet.sfsnet.load_state_dict(checkpoint['sfsnet_state_dict'])
    print(checkpoint_path, 'loaded')
    for param in mvsfsnet.sfsnet.parameters():
        param.requires_grad = False
mvsfsnet = nn.DataParallel(mvsfsnet)
mvsfsnet.to(device)

depth_loss = DepthLogLikelihoodLoss()
dn_consistency_loss = DepthNormalConsistencyLoss()

# prepare Gaussian kernel
def create_gaussian_kernel(l=5, sigma=1.0):
    x = np.arange(2*l+1) - l
    f = np.exp(-0.5 * x**2 / sigma**2)
    kernel = f[:,None] @ f[None,:]

    return kernel / np.sum(kernel)

gaussian_kernel = create_gaussian_kernel(10, 3.0).astype(np.float32)
gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)[None,None].repeat(3,1,1,1)

def compute_depth_error(pred_depth, gt_depth, mask, depth_values):
    depth_range = depth_values[:,-1] - depth_values[:,0]
    error_map = torch.abs(pred_depth-gt_depth) * mask / depth_range[:,None,None,None]
    return torch.sum(error_map) / torch.sum(mask)


def normal_loss(pred_normal, gt_normal, mask):
    cosine = torch.sum(pred_normal * gt_normal, dim=1)
    angle_error = torch.acos(torch.clamp(cosine, -0.9999, 0.9999)) * mask[:,0]
    return torch.sum(angle_error) / torch.sum(mask[:,0])

# optimizer
optimizer = torch.optim.Adam(mvsfsnet.parameters(), weight_decay=1e-5)

list_ckpt = sorted(glob.glob(weight_dir+'/???.ckpt'))
idx_itr_ofs = 0
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mvsfsnet.module.load_state_dict(checkpoint['mvsfsnet_state_dict'])
    idx_itr_ofs = len(list_ckpt)

for idx_itr in range(idx_itr_ofs, 30):
    bar = tqdm(trainloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (train)')
    mvsfsnet.train()
    if not args.wo_sfs:
        mvsfsnet.module.sfsnet.eval()
    total_depth_loss = 0.0
    total_normal_loss = 0.0
    total_dn_loss = 0.0
    total_sfs_loss = 0.0
    for idx_minbatch, minbatch in enumerate(bar):
        imgs = minbatch['imgs'].to(device)
        rmaps = minbatch['rmaps'].to(device)
        intrinsics = minbatch['intrinsics'].to(device)
        proj_matrices = minbatch['proj_matrices'].to(device)
        rot_matrices = minbatch['rot_matrices'].to(device)
        depth_values = minbatch['depth_values'].to(device)
        gt_depth = minbatch['depths'][:,0].to(device)
        gt_normal = minbatch['normals'][:,0].to(device)

        masks = (minbatch['depths'] > 0.0).float().to(device)

        # ignore pixels whose gt depth is not within the range of depth_values
        masks[:,0] = (gt_depth > depth_values[:,0:1,None,None]).float() * masks[:,0]
        masks[:,0] = (gt_depth < depth_values[:,-1:,None,None]).float() * masks[:,0]

        out = mvsfsnet(imgs, rmaps, rot_matrices, proj_matrices, depth_values)
        #depth_prob_volume = out['depth_prob_volume']
        normal_volume = out['normal_volume']
        pred_depth = out['depth'][:,None] * masks[:,0]
        pred_normal = out['normal'] * masks[:,0]
        depth_prob_volume = out['depth_prob_volume'] * masks[:,0,0,None]
        normal_volume = out['normal_volume'] * masks[:,0,:,None]

        # make log-radiance volumes
        imgs[torch.isnan(imgs)] = 0.0
        imgs[torch.isinf(imgs)] = 0.0

        # make log-radiance volumes
        img_volumes = construct_image_feature_volumes(imgs, proj_matrices, depth_values) # [BS,N,3,D,H,W]
        log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None)) # [BS,N,3,D,H,W]

        # make warped src images
        warped_log_imgs = torch.sum(log_img_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]

        # make log-radiance volumes
        rmap_volumes = construct_reflectance_feature_volumes(rmaps, rot_matrices, normal_volume) # [BS,N,3,D,H,W]
        log_rmap_volumes = torch.log1p(torch.clamp(1000 * rmap_volumes, 0, None)) # [BS,N,3,D,H,W]

        # make rendered log images
        rendered_log_imgs = torch.sum(log_rmap_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]

        warped_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(warped_log_imgs, dim=1)], dim=1)
        rendered_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(rendered_log_imgs, dim=1)], dim=1)

        # create blurred log images
        log_error_imgs_blur = torch.mean(torch.abs(warped_log_imgs_blurred - rendered_log_imgs_blurred), dim=2, keepdim=True) # [BS,N,1,H,W]
        log_error_img_blur = torch.mean(log_error_imgs_blur, dim=1) # [BS,1,H,W]

        normal_from_depth = depth_to_normal(pred_depth[:,0], intrinsics[:,0]) * masks[:,0]
        #normal_mask = (torch.sum(normal_from_depth**2, dim=1, keepdim=True) > 0.0).float()

        #radiance_from_imgs = sample_from_src_views(pred_depth, imgs, proj_matrices)[0]
        #log_radiance_from_imgs = torch.log1p(torch.clamp(1000 * radiance_from_imgs, 0, None))
        #radiance_from_rmaps = construct_reflectance_feature_volumes(rmaps, rot_matrices, pred_normal[:,:,None])[:,:,:,0] * masks[:,0:1]
        #log_radiance_from_rmaps = torch.log1p(torch.clamp(1000 * radiance_from_rmaps, 0, None))

        #log_error_maps = torch.mean(torch.abs(log_radiance_from_imgs - log_radiance_from_rmaps), dim=2)
        #log_error_map = torch.mean(log_error_maps, dim=1) * masks[:,0,0]

        loss_d = compute_depth_error(pred_depth, gt_depth, masks[:,0], depth_values)
        loss_n = normal_loss(pred_normal, gt_normal, masks[:,0])
        loss_dn = normal_loss(pred_normal, normal_from_depth, masks[:,0])
        loss_shading = torch.sum(log_error_img_blur * masks[:,0]) / torch.sum(masks[:,0])
        loss = 100 * loss_d + loss_n + 0.1 * loss_dn + 0.0 * loss_shading

        # backward
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # update bar postfix
        total_depth_loss += loss_d.item()
        total_normal_loss += loss_n.item()
        total_dn_loss += loss_dn.item()
        total_sfs_loss += loss_shading.item()
        bar.set_postfix(
            depth_loss=total_depth_loss/(idx_minbatch+1),
            normal_loss=total_normal_loss/(idx_minbatch+1),
            dn_loss=total_dn_loss/(idx_minbatch+1),
            sfs_loss=total_sfs_loss/(idx_minbatch+1)
        )
    train_depth_loss = total_depth_loss/(idx_minbatch+1)
    train_normal_loss = total_normal_loss/(idx_minbatch+1)
    train_dn_loss = total_dn_loss/(idx_minbatch+1)
    train_sfs_loss = total_sfs_loss/(idx_minbatch+1)

    bar = tqdm(valloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (val)')
    mvsfsnet.eval()
    total_depth_loss = 0.0
    total_normal_loss = 0.0
    total_dn_loss = 0.0
    total_sfs_loss = 0.0
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            imgs = minbatch['imgs'].to(device)
            rmaps = minbatch['rmaps'].to(device)
            intrinsics = minbatch['intrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            rot_matrices = minbatch['rot_matrices'].to(device)
            depth_values = minbatch['depth_values'].to(device)
            gt_depth = minbatch['depths'][:,0].to(device)
            gt_normal = minbatch['normals'][:,0].to(device)

            masks = (minbatch['depths'] > 0.0).float().to(device)

            # ignore pixels whose gt depth is not within the range of depth_values
            masks[:,0] = (gt_depth > depth_values[:,0:1,None,None]).float() * masks[:,0]
            masks[:,0] = (gt_depth < depth_values[:,-1:,None,None]).float() * masks[:,0]

            out = mvsfsnet(imgs, rmaps, rot_matrices, proj_matrices, depth_values)
            #depth_prob_volume = out['depth_prob_volume']
            normal_volume = out['normal_volume']
            pred_depth = out['depth'][:,None] * masks[:,0]
            pred_normal = out['normal'] * masks[:,0]
            depth_prob_volume = out['depth_prob_volume'] * masks[:,0,0,None]
            normal_volume = out['normal_volume'] * masks[:,0,:,None]

            # make log-radiance volumes
            imgs[torch.isnan(imgs)] = 0.0
            imgs[torch.isinf(imgs)] = 0.0

            # make log-radiance volumes
            img_volumes = construct_image_feature_volumes(imgs, proj_matrices, depth_values) # [BS,N,3,D,H,W]
            log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None)) # [BS,N,3,D,H,W]

            # make warped src images
            warped_log_imgs = torch.sum(log_img_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]

            # make log-radiance volumes
            rmap_volumes = construct_reflectance_feature_volumes(rmaps, rot_matrices, normal_volume) # [BS,N,3,D,H,W]
            log_rmap_volumes = torch.log1p(torch.clamp(1000 * rmap_volumes, 0, None)) # [BS,N,3,D,H,W]

            # make rendered log images
            rendered_log_imgs = torch.sum(log_rmap_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]

            warped_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(warped_log_imgs, dim=1)], dim=1)
            rendered_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(rendered_log_imgs, dim=1)], dim=1)

            # create blurred log images
            log_error_imgs_blur = torch.mean(torch.abs(warped_log_imgs_blurred - rendered_log_imgs_blurred), dim=2, keepdim=True) # [BS,N,1,H,W]
            log_error_img_blur = torch.mean(log_error_imgs_blur, dim=1) # [BS,1,H,W]

            normal_from_depth = depth_to_normal(pred_depth[:,0], intrinsics[:,0]) * masks[:,0]
            #normal_mask = (torch.sum(normal_from_depth**2, dim=1, keepdim=True) > 0.0).float()

            #radiance_from_imgs = sample_from_src_views(pred_depth, imgs, proj_matrices)[0]
            #log_radiance_from_imgs = torch.log1p(torch.clamp(1000 * radiance_from_imgs, 0, None))
            #radiance_from_rmaps = construct_reflectance_feature_volumes(rmaps, rot_matrices, pred_normal[:,:,None])[:,:,:,0] * masks[:,0:1]
            #log_radiance_from_rmaps = torch.log1p(torch.clamp(1000 * radiance_from_rmaps, 0, None))

            #log_error_maps = torch.mean(torch.abs(log_radiance_from_imgs - log_radiance_from_rmaps), dim=2)
            #log_error_map = torch.mean(log_error_maps, dim=1) * masks[:,0,0]

            loss_d = compute_depth_error(pred_depth, gt_depth, masks[:,0], depth_values)
            loss_n = normal_loss(pred_normal, gt_normal, masks[:,0])
            loss_dn = normal_loss(pred_normal, normal_from_depth, masks[:,0])
            loss_shading = torch.sum(log_error_img_blur * masks[:,0]) / torch.sum(masks[:,0])
            loss = 100 * loss_d + loss_n + 0.1 * loss_dn + 0.0 * loss_shading

            # update bar postfix
            total_depth_loss += loss_d.item()
            total_normal_loss += loss_n.item()
            total_dn_loss += loss_dn.item()
            total_sfs_loss += loss_shading.item()
            bar.set_postfix(
                depth_loss=total_depth_loss/(idx_minbatch+1),
                normal_loss=total_normal_loss/(idx_minbatch+1),
                dn_loss=total_dn_loss/(idx_minbatch+1),
                sfs_loss=total_sfs_loss/(idx_minbatch+1)
            )
    val_depth_loss = total_depth_loss/(idx_minbatch+1)
    val_normal_loss = total_normal_loss/(idx_minbatch+1)
    val_dn_loss = total_dn_loss/(idx_minbatch+1)
    val_sfs_loss = total_sfs_loss/(idx_minbatch+1)

    # save weights
    torch.save({
            'mvsfsnet_state_dict': mvsfsnet.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_normal_loss': train_normal_loss,
            'train_depth_loss': train_depth_loss,
            'train_dn_loss': train_dn_loss,
            'train_sfs_loss': train_sfs_loss,
            'val_normal_loss': val_normal_loss,
            'val_depth_loss': val_depth_loss,
            'val_dn_loss': val_dn_loss,
            'val_sfs_loss': val_sfs_loss,
    }, weight_dir+'/'+str(idx_itr).zfill(3)+'.ckpt') 