import os

os.environ["MPLBACKEND"]="WebAgg"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.dataset import MVSRDataset
from core.mvs_models import MVSfSNet
from core.sfs_utils import *
from core.mvs_utils import get_depth_values, depth_from_silhouette, construct_reflectance_feature_volumes, refine_normal_volume, construct_image_feature_volumes, depth_to_normal, warp_normal_maps, warp_depth_maps
from core.mvs_criterion import DepthNormalConsistencyLoss

from core.ibrdf import IBRDF
from core.ibrdf.renderer import Renderer
from core.ibrdf.util import SaveMERL, SaveHDR
from core.rendering_utils import decode_brdf, backward_brdf, render_sphere

from eval_utils.visualize_merl import visualize_merl_as_sheres

import numpy as np
import matplotlib.pyplot as plt

import json

import sys
from tqdm import tqdm
import glob

def create_gaussian_kernel(l=5, sigma=1.0):
    x = np.arange(2*l+1) - l
    f = np.exp(-0.5 * x**2 / sigma**2)
    kernel = f[:,None] @ f[None,:]

    return kernel / np.sum(kernel)

def compute_depth_error(pred_depth, gt_depth, mask, depth_values):
    mask = mask * (gt_depth > 0.0).float()
    depth_range = depth_values[:,-1] - depth_values[:,0]
    error_map = torch.abs(pred_depth-gt_depth) * mask / depth_range[:,None,None,None]
    return torch.sum(error_map) / torch.sum(mask)


def compute_normal_error(pred_normal, gt_normal, mask):
    mask = mask * (torch.sum(gt_normal**2, dim=1, keepdim=True) > 0.0).float()
    cosine = torch.sum(pred_normal * gt_normal, dim=1)
    angle_error = torch.acos(torch.clamp(cosine, -0.9999, 0.9999)) * mask[:,0]
    return torch.sum(angle_error) / torch.sum(mask[:,0])

def est_shape_mat(
    test_dataset, 
    out_dir,
    device,
    num_views_for_rfs = 3,
    numdepth=192,
    beta = 0.0,
    occlusion_handling = True,
    bbox_diagonal = None,
    threshold_final_depth_error = 0.10,
    threshold_final_normal_error = 0.5235987755982988,
    num_epoch = 40,
    num_iteration_per_epoch = 5,
    wo_sfs = False,
    wo_coarse = False,
    wo_fine = False,
    spp_final = 8192,
    est_depth_range_list = None
):
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    testloader_shuffled = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # estimate depth range from silhouettes
    if est_depth_range_list is None:
        src_masks = []
        src_projs = []
        for idx_view, minbatch in enumerate(testloader):
            # load images, illumination map, and camera parameters
            src_mask = minbatch['masks'].to(device)[:,0]
            src_proj = minbatch['proj_matrices'].to(device)[:,0]
            src_masks.append(src_mask)
            src_projs.append(src_proj)
        src_masks = torch.stack(src_masks, dim=1)
        src_projs = torch.stack(src_projs, dim=1)

        bar = tqdm(testloader)
        bar.set_description('Coarse Depth from Silhouette')
        est_depth_range_list = []
        for idx_view, minbatch in enumerate(bar):
            # load images, illumination map, and camera parameters
            ref_mask = minbatch['masks'].to(device)[:,0]
            ref_intrinsic = minbatch['intrinsics'].to(device)[:,0]
            ref_proj = minbatch['proj_matrices'].to(device)[:,0]
            ref_depth_range = minbatch['depth_ranges'].to(device)[:,0]
            depth_from_mask = depth_from_silhouette(ref_mask, ref_proj, ref_intrinsic, ref_depth_range, src_masks, src_projs, numdepth=4*numdepth)[0]
            dmax = torch.max(depth_from_mask.view(depth_from_mask.size(0),-1), dim=1)[0]
            depth_from_mask[depth_from_mask == 0.0] = 1e12
            dmin = torch.min(depth_from_mask.view(depth_from_mask.size(0),-1), dim=1)[0]
            est_depth_range = torch.cat([dmin, dmax])

            est_depth_range_list.append(est_depth_range.cpu())

    # load model
    mvsfsnet = MVSfSNet(wo_sfs=wo_sfs)
    val_loss_min = 1e12
    checkpoint_path = None
    weight_dir = './weights/nlmvsnet'
    if wo_sfs:
        weight_dir = weight_dir+'-wo-sfs'
    for f in sorted(glob.glob(weight_dir+'/*.ckpt')):
        val_loss = torch.load(f)['val_normal_loss']
        if val_loss < val_loss_min:
            checkpoint_path = f
            val_loss_min = val_loss
    checkpoint = torch.load(checkpoint_path)
    mvsfsnet.load_state_dict(checkpoint['mvsfsnet_state_dict'])
    print(checkpoint_path, 'loaded')
    for p in mvsfsnet.parameters():
        p.requires_grad = False
    mvsfsnet.eval()
    mvsfsnet.to(device)

    # load ibrdf model and renderer
    numLayers = 6
    numInputFeatures = 3
    numEmbedDim = 16
    numPiecesPerLayer = 8

    ibrdf = IBRDF(numLayers, numInputFeatures, numEmbedDim, numPiecesPerLayer)
    state_dict = torch.load('./data/ibrdf.pt')
    ibrdf.load_state_dict(state_dict)
    for p in ibrdf.parameters():
        p.requires_grad = False
    ibrdf.eval()
    ibrdf.to(device)

    renderer = Renderer(use_importance_sampling=True)

    dn_consistency_loss = DepthNormalConsistencyLoss()

    gaussian_kernel = create_gaussian_kernel(10, 3.0).astype(np.float32)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)[None,None].repeat(3,1,1,1)

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=beta)

    # initialize brdf parameters
    log_color = torch.zeros((1,3)).to(device)
    embed_code= torch.zeros((1,3, numEmbedDim)).to(device)
    with torch.no_grad():
        embed_code.requires_grad = True
        log_color.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': embed_code, 'lr': 1e-2}, 
        {'params': log_color, 'lr': 2e-3}, 
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    idx_epoch_0 = 0
    loss_list = []
    dn_error_list = []
    depth_error_list = []
    normal_error_list = []
    if os.path.exists(out_dir+'/result_tmp.pt'):
        checkpoint = torch.load(out_dir+'/result_tmp.pt')
        print('existing checkpoint loaded')
        with torch.no_grad():
            idx_epoch_0 = checkpoint['idx_epoch'] + 1
            embed_code[:] = checkpoint['embed_code'].to(device)
            log_color[:] = checkpoint['log_color'].to(device)
        loss_list = checkpoint['loss_list']
        dn_error_list = checkpoint['dn_error_list']
        depth_error_list = checkpoint['depth_error_list']
        normal_error_list = checkpoint['normal_error_list']
        optimizer.load_state_dict(checkpoint['oprimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        result = checkpoint

    for idx_epoch in range(idx_epoch_0, num_epoch):
        if wo_sfs:
            break

        # shape from reflectance
        # decode brdf
        with torch.no_grad():
            brdf = decode_brdf(ibrdf, embed_code, log_color)

        # estimate shape for each view
        list_depth_prob_volume = []
        list_normal_volume = []
        list_depth_values = []
        list_est_depth = []
        list_est_normal = []
        list_est_rmaps = []
        list_warped_log_imgs = []
        total_depth_error = 0.0
        total_normal_error = 0.0
        total_dn_error = 0.0
        bar = tqdm(testloader)
        bar.set_description('Epoch '+str(1+idx_epoch).zfill(2)+' SfR')
        rendered_rmaps = [None for i in range(max(50,len(testloader)))]
        for idx_view, minbatch in enumerate(bar):
            # load images, illumination map, and camera parameters
            imgs = minbatch['hdr_images'].to(device)
            masks = minbatch['masks'].to(device)
            illum_map = minbatch['illum_map'].to(device)
            intrinsics = minbatch['intrinsics'].to(device)
            extrinsics = minbatch['extrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            depth_ranges = minbatch['depth_ranges'].to(device)
            view_indices = minbatch['view_indices'].to(device)

            rot_matrices = extrinsics[:,:,:3,:3]

            illum_map = F.interpolate(illum_map, (256,512), mode='area')

            # load gt data
            gt_depth = None if not 'gt_depths' in minbatch else minbatch['gt_depths'].to(device)[:,0]
            gt_normal = None if not 'gt_normals' in minbatch else minbatch['gt_normals'].to(device)[:,0]
            gt_rmaps = None if not 'hdr_rmaps' in minbatch else minbatch['hdr_rmaps'].to(device)

            # make sure that the gt normal is normalized
            if not (gt_normal is None):
                l = torch.sqrt(torch.clamp(torch.sum(gt_normal**2, dim=1, keepdim=True), 1e-2, None))
                gt_normal = (gt_normal / l)

            # compute depth values
            depth_values = get_depth_values(
                torch.mean(depth_ranges[:,0], dim=1), 
                imgs.size()[3:5], 
                intrinsics[:,0], 
                numdepth=numdepth
            )

            # adjust depth_values to gt depth range
            est_depth_range = est_depth_range_list[idx_view].to(device)
            center_gt_depth = torch.exp(0.5 * (torch.log(est_depth_range[0]) + torch.log(est_depth_range[1])))
            center_depth_values = torch.exp(0.5 * (torch.log(depth_values[:,0]) + torch.log(depth_values[:,-1])))
            depth_values *= center_gt_depth / center_depth_values

            # render rmaps
            est_rmaps = []
            for i in range(extrinsics.size(1)):
                if rendered_rmaps[view_indices[0,i]] is None:
                    est_rmap = render_sphere(renderer, brdf, illum_map, extrinsics[:,i], spp=8192)
                    rendered_rmaps[view_indices[0,i]] = est_rmap
                else:
                    est_rmap = rendered_rmaps[view_indices[0,i]]
                est_rmaps.append(est_rmap)
            est_rmaps = torch.stack(est_rmaps, dim=1)

            # make rmap_masks
            v,u = torch.meshgrid(torch.arange(est_rmaps.size(3)), torch.arange(est_rmaps.size(4)))
            x = 2 * (u + 0.5) / 128 - 1
            y = -(2 * (v + 0.5) / 128 - 1)
            z = torch.sqrt(torch.clamp(1-x**2-y**2,0,None))
            rmap_mask = (z > 0.0).float()[None,None].to(device)
            rmap_masks = rmap_mask[:,None,:,:,:].repeat(1,est_rmaps.size(1),1,1,1)

            if (idx_epoch == 0):
                # roughly adjust color
                ci = torch.sum(imgs * masks, dim=(0,1,3,4)) / torch.sum(masks, dim=(0,1,3,4))
                cr = torch.sum(est_rmaps * rmap_masks, dim=(0,1,3,4)) / torch.sum(rmap_masks, dim=(0,1,3,4))
                est_rmaps *= (ci / cr)[None,:,None,None]

                with torch.no_grad():
                    log_color[:] += torch.log(ci/cr) / len(testloader)

            # estimate shape as volumes
            out = mvsfsnet(imgs, est_rmaps.detach(), rot_matrices, proj_matrices, depth_values)
            est_depth = out['depth'][:,None] * masks[:,0]
            est_normal = out['normal'] * masks[:,0]
            depth_prob_volume = out['depth_prob_volume'] * masks[:,0,0,None]
            normal_volume = out['normal_volume'] * masks[:,0,:,None]

            # evaluate depth-normal consistency
            dn_error = dn_consistency_loss(depth_prob_volume, normal_volume, masks[:,0], intrinsics[:,0], depth_values)

            # compute normal / depth errors
            depth_error = compute_depth_error(est_depth, gt_depth, masks[:,0], depth_values)
            normal_error = compute_normal_error(est_normal, gt_normal, masks[:,0])

            # make warped src images
            img_volumes = construct_image_feature_volumes(imgs, proj_matrices, depth_values) # [BS,N,3,D,H,W]
            log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None)) # [BS,N,3,D,H,W]
            warped_log_imgs = torch.sum(log_img_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]
            #warped_imgs = sample_from_src_views(est_depth, imgs, proj_matrices)[0] * masks[:,0:1]
            #warped_log_imgs = torch.log1p(torch.clamp(1000 * warped_imgs, 0, None)) # [BS,N,3,H,W]

            list_depth_prob_volume.append(depth_prob_volume.cpu())
            list_normal_volume.append(normal_volume.cpu())
            list_depth_values.append(depth_values.cpu())
            list_est_depth.append(est_depth.cpu())
            list_est_normal.append(est_normal.cpu())
            list_est_rmaps.append(est_rmaps.cpu())
            list_warped_log_imgs.append(warped_log_imgs.cpu())
            total_depth_error += depth_error.item()
            total_normal_error += normal_error.item()
            total_dn_error += dn_error.item()
            bar.set_postfix(
                mean_depth_error = total_depth_error/(idx_view+1),
                mean_normal_error = total_normal_error/(idx_view+1),
                mean_dn_error = total_dn_error/(idx_view+1),
            )

        # save average error to list
        depth_error_list.append(total_depth_error / len(testloader))
        normal_error_list.append(total_normal_error / len(testloader))
        dn_error_list.append(total_dn_error / len(testloader))

        del rendered_rmaps

        # compute occlusion masks
        list_occlusion_masks = []
        bar = tqdm(testloader)
        bar.set_description('Epoch '+str(1+idx_epoch).zfill(2)+' Occlusion')
        for idx_view, minbatch in enumerate(bar):
            # load images, illumination map, and camera parameters
            masks = minbatch['masks'].to(device)
            extrinsics = minbatch['extrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            view_indices = minbatch['view_indices'].to(device)

            if occlusion_handling == False:
                occlusion_masks = torch.zeros_like(masks[:,0:1], dtype=masks.dtype, device=device)
                occlusion_masks = occlusion_masks.repeat(1,masks.size(1),1,1,1)
                occlusion_masks[:,:num_views_for_rfs] = masks[:,0:1]
            else:
                rot_matrices = extrinsics[:,:,:3,:3]

                # load shape estimation results
                est_depth = list_est_depth[idx_view].to(device)
                est_depths = torch.stack([list_est_depth[i].to(device) for i in view_indices[0]], dim=1)
                est_normals = torch.stack([list_est_normal[i].to(device) for i in view_indices[0]], dim=1)

                est_depths_warped, est_depths_sampled = warp_depth_maps(est_depths, proj_matrices, est_depth)
                est_normals_warped = warp_normal_maps(est_normals, rot_matrices, proj_matrices, est_depth)

                est_depths_error = torch.abs(est_depths_sampled - est_depths_warped)
                est_normals_cosine = torch.sum(est_normals[:,0:1] * est_normals_warped, dim=2, keepdim=True)
                #est_normals_cosine = torch.clamp(est_normals_cosine,0,1) * (est_normals_warped[:,:,2:3] > 0.0).float()
                visibility_scores = torch.exp(-est_depths_error) * (est_normals_cosine > 0.0).float() * (est_normals_warped[:,:,2:3] > 0.0).float()

                sorted_indices = torch.argsort(visibility_scores,dim=1,descending=True) # [BS,N,1,H,W]
                idx_array = torch.arange(est_normals.size(1), dtype=int, device=device)
                occlusion_masks = torch.any(
                    idx_array[None,:,None,None,None,None] == sorted_indices[:,None,:num_views_for_rfs], 
                    dim=2
                ).float() * masks[:,0:1] # [BS,N,1,H,W]

            list_occlusion_masks.append(occlusion_masks.cpu())

        # compute refined normal volume
        list_refined_normal_volume = []
        bar = tqdm(testloader)
        bar.set_description('Epoch '+str(1+idx_epoch).zfill(2)+' NV Refinement')
        for idx_view, minbatch in enumerate(bar):
            # load images, illumination map, and camera parameters
            imgs = minbatch['hdr_images'].to(device)
            masks = minbatch['masks'].to(device)
            extrinsics = minbatch['extrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            view_indices = minbatch['view_indices'].to(device)

            rot_matrices = extrinsics[:,:,:3,:3]

            est_rmaps = list_est_rmaps[idx_view].to(device)
            normal_volume = list_normal_volume[idx_view].to(device)
            occlusion_masks = list_occlusion_masks[idx_view].to(device)
            depth_values = list_depth_values[idx_view].to(device)

            refined_normal_volume = refine_normal_volume(
                imgs, 
                est_rmaps, 
                normal_volume, 
                occlusion_masks,
                masks[:,0], 
                proj_matrices, 
                rot_matrices, 
                depth_values
            )

            list_refined_normal_volume.append(refined_normal_volume.cpu())
        
        # reflectance from shape
        running_loss_list = []
        if len(test_dataset) == 1:
            bar = tqdm(range(num_iteration_per_epoch))
            bar.set_description('Epoch '+str(1+idx_epoch).zfill(2)+' RfS')
            range_itr = bar
        else:
            range_itr = range(num_iteration_per_epoch) 
        for idx_itr in range_itr:
            total_loss = 0.0
            if len(test_dataset) != 1:
                bar = tqdm(testloader_shuffled)
                bar.set_description('Epoch '+str(1+idx_epoch).zfill(2)+' RfS(itr='+str(idx_itr).zfill(2)+')')
                loader = bar
            else:
                loader = testloader_shuffled
            #log_colors = []
            for idx_batch, minbatch in enumerate(loader):
                if len(test_dataset) == 1:
                    idx_view = 0
                else:
                    idx_view = minbatch['view_indices'][0,0].item()
                # load images, illumination map, and camera parameters
                imgs = minbatch['hdr_images'].to(device)
                masks = minbatch['masks'].to(device)
                illum_map = minbatch['illum_map'].to(device)
                intrinsics = minbatch['intrinsics'].to(device)
                extrinsics = minbatch['extrinsics'].to(device)
                proj_matrices = minbatch['proj_matrices'].to(device)
                depth_ranges = minbatch['depth_ranges'].to(device)

                illum_map = F.interpolate(illum_map, (256,512), mode='area')

                # load gt data
                gt_depth = None if not 'gt_depths' in minbatch else minbatch['gt_depths'].to(device)[:,0]
                gt_normal = None if not 'gt_normals' in minbatch else minbatch['gt_normals'].to(device)[:,0]
                gt_rmaps = None if not 'hdr_rmaps' in minbatch else minbatch['hdr_rmaps'].to(device)

                if occlusion_handling == False:
                    imgs = imgs[:,:num_views_for_rfs]
                    masks = masks[:,:num_views_for_rfs]
                    intrinsics = intrinsics[:,:num_views_for_rfs]
                    extrinsics = extrinsics[:,:num_views_for_rfs]
                    proj_matrices = proj_matrices[:,:num_views_for_rfs]
                    depth_ranges = depth_ranges[:,:num_views_for_rfs]
                    if not (gt_rmaps is None):
                        gt_rmaps = gt_rmaps[:,:num_views_for_rfs]

                rot_matrices = extrinsics[:,:,:3,:3]

                # make sure that the gt normal is normalized
                if not (gt_normal is None):
                    l = torch.sqrt(torch.clamp(torch.sum(gt_normal**2, dim=1, keepdim=True), 1e-2, None))
                    gt_normal = (gt_normal / l)

                # load shape estimation results
                depth_prob_volume = list_depth_prob_volume[idx_view].to(device)
                normal_volume = list_normal_volume[idx_view].to(device)
                refined_normal_volume = list_refined_normal_volume[idx_view].to(device)
                occlusion_masks = list_occlusion_masks[idx_view].to(device)
                est_depth = list_est_depth[idx_view].to(device)
                est_normal = list_est_normal[idx_view].to(device)
                warped_log_imgs = list_warped_log_imgs[idx_view].to(device)
                depth_values = list_depth_values[idx_view].to(device)

                if occlusion_handling == False:
                    occlusion_masks = occlusion_masks[:,:num_views_for_rfs]
                    warped_log_imgs = warped_log_imgs[:,:num_views_for_rfs]

                # decode brdf
                with torch.no_grad():
                    brdf = decode_brdf(ibrdf, embed_code, log_color)
                    brdf.requires_grad = True

                # render rmaps
                est_rmaps = render_sphere(
                    renderer, 
                    brdf.repeat(extrinsics.size(1),1,1,1,1), # [BS*N,3,90,90,180]
                    illum_map.repeat(extrinsics.size(1),1,1,1), # [BS*N,3,He,We]
                    extrinsics.view(-1,4,4) # [BS*N,4,4]
                ).view(extrinsics.size(0), extrinsics.size(1), 3, 128, 128)
                est_log_rmaps = torch.log1p(torch.clamp(1000 * est_rmaps, 0, None))

                # naive image reconstruction
                rmap_volumes = construct_reflectance_feature_volumes(est_rmaps, rot_matrices, normal_volume) # [BS,N,3,D,H,W]
                log_rmap_volumes = torch.log1p(torch.clamp(1000 * rmap_volumes, 0, None)) # [BS,N,3,D,H,W]
                est_warped_log_imgs_naive = torch.sum(log_rmap_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]
                #est_warped_imgs_naive = construct_reflectance_feature_volumes(est_rmaps, rot_matrices, est_normal[:,:,None])[:,:,:,0] * masks[:,0:1,:,:,:] # [BS,N,3,H,W]
                #est_warped_log_imgs_naive = torch.log1p(torch.clamp(1000 * est_warped_imgs_naive, 0, None)) # [BS,N,3,H,W]

                # coarse consistency
                warped_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(warped_log_imgs, dim=1)], dim=1)
                est_warped_log_imgs_blurred = torch.stack([F.conv2d(log_img, gaussian_kernel, groups=3, padding=(gaussian_kernel.size(2) // 2)) for log_img in torch.unbind(est_warped_log_imgs_naive, dim=1)], dim=1)
                log_img_errors_coarse = torch.mean(smooth_l1_loss(est_warped_log_imgs_blurred, warped_log_imgs_blurred), dim=2, keepdim=True) * masks[:,0:1] # [BS,N,1,H,W]
                log_img_error_coarse = torch.sum(log_img_errors_coarse * occlusion_masks, dim=1) / torch.clamp(torch.sum(occlusion_masks, dim=1), 1e-3, None)
                loss_coarse = torch.sum(log_img_error_coarse * masks[:,0]) / torch.sum(masks[:,0])

                # fine consistency
                refined_rmap_volumes = construct_reflectance_feature_volumes(est_rmaps, rot_matrices, refined_normal_volume) # [BS,N,3,D,H,W]
                refined_log_rmap_volumes = torch.log1p(torch.clamp(1000 * refined_rmap_volumes, 0, None)) # [BS,N,3,D,H,W]
                est_warped_log_imgs_refined = torch.sum(refined_log_rmap_volumes * depth_prob_volume[:,None,None,:,:,:], dim=3) * masks[:,0:1] # [BS,N,3,H,W]

                log_img_errors_fine = torch.mean(smooth_l1_loss(est_warped_log_imgs_refined, warped_log_imgs), dim=2, keepdim=True) * masks[:,0:1] # [BS,N,1,H,W]
                log_img_error_fine = torch.sum(log_img_errors_fine * occlusion_masks, dim=1) / torch.clamp(torch.sum(occlusion_masks, dim=1), 1e-3, None)
                loss_fine = torch.sum(log_img_error_fine * masks[:,0]) / torch.sum(masks[:,0])

                log_img_error = 0.5 * (log_img_error_fine + log_img_error_coarse)
                if wo_coarse and wo_fine:
                    log_img_errors_naive = torch.mean(smooth_l1_loss(est_warped_log_imgs_naive, warped_log_imgs), dim=2, keepdim=True) * masks[:,0:1] # [BS,N,1,H,W]
                    log_img_error_naive = torch.sum(log_img_errors_naive * occlusion_masks, dim=1) / torch.clamp(torch.sum(occlusion_masks, dim=1), 1e-3, None)
                    loss = torch.sum(log_img_error_naive * masks[:,0]) / torch.sum(masks[:,0])
                elif wo_coarse:
                    loss = loss_fine
                elif wo_fine:
                    loss = loss_coarse
                else:
                    loss = 0.5 * (loss_fine + loss_coarse)

                # reconstruct the reference image
                recon_ref_img = (torch.expm1(est_warped_log_imgs_naive[:,0]) / 1000)

                # compute reconstruction loss
                error_map = log_img_error

                # backward to brdf array
                loss.backward()
                with torch.no_grad():
                    grad_brdf = brdf.grad

                # backward to the brdf code
                optimizer.zero_grad()
                backward_brdf(ibrdf, embed_code, log_color, grad_brdf, ChunkSize=50000)

                # update brdf code
                optimizer.step()

                del grad_brdf

                # make embed code inside the unit supersphere
                with torch.no_grad():
                    code_norm = torch.sqrt(torch.sum(embed_code**2, dim=2, keepdim=True))
                    embed_code /= torch.clamp(code_norm, 1.0, None)

                total_loss += loss.item()
                #log_colors.append(log_color)
                bar.set_postfix(
                    mean_loss = total_loss/(idx_batch+1),
                )
            #log_color = torch.mean(torch.stack(log_colors, dim=0), dim=0)

            # save temporal result for each iteration
            running_loss_list.append(total_loss / (idx_batch+1))
            plt.plot(running_loss_list)
            plt.grid()
            plt.xlabel('iteration')
            plt.ylabel('image reconstruction loss')
            plt.savefig(out_dir+'/loss_epoch_'+str(idx_epoch).zfill(3)+'.png')
            plt.close()

            plt.subplot(6,5,1)
            plot_hdr(imgs[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,2)
            plot_hdr(est_rmaps[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,3)
            plot_normal_map(est_normal)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,4)
            plt.imshow(est_depth[0,0].detach().cpu().numpy(), vmin=depth_values[0,0].item(), vmax=depth_values[0,-1].item())
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,5)
            plot_hdr(recon_ref_img / torch.clamp(torch.max(imgs), None, 1.0))
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,8)
            plot_normal_map(normal_volume[:,:,:,64,:])
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(6,5,9)
            plt.imshow(depth_prob_volume[0,:,64,:].detach().cpu().numpy())
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            #plt.subplot(6,5,10)
            #plt.imshow(torch.exp(-log_error_volume.detach()[0,:,64,:]).cpu().numpy(), vmin=0, vmax=None)
            #plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            #plt.tick_params(bottom=False,left=False,right=False,top=False)
            #plt.box(False)

            for i in range(min(5,warped_log_imgs.size(1))):
                plt.subplot(6,5,5*2+1+i)
                plot_hdr((torch.expm1(warped_log_imgs[:,i]) / 1000))
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.subplot(6,5,5*3+1+i)
                plot_hdr((torch.expm1(warped_log_imgs_blurred[:,i]) / 1000))
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.subplot(6,5,5*4+1+i)
                plot_hdr((torch.expm1(est_warped_log_imgs_blurred[:,i]) / 1000))
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.subplot(6,5,5*5+1+i)
                plot_hdr((torch.expm1(est_warped_log_imgs_naive[:,i]) / 1000))
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)

            plt.savefig(out_dir+'/tmp.svg')
            plt.close()

            os.makedirs(out_dir+'/tmp', exist_ok=True)
            cv2.imwrite(out_dir+'/tmp/depth.png', (255 * torch.clamp((est_depth[0,0] - depth_values[0,0]) / (depth_values[0,-1] - depth_values[0,0]), 0, 1)).detach().cpu().numpy().astype(np.uint8))
            cv2.imwrite(out_dir+'/tmp/normal.png', (255 * np.clip(0.5*(est_normal[0].detach().cpu().numpy().transpose(1,2,0) + 1.0), 0, 1)).astype(np.uint8)[:,:,::-1])
            #cv2.imwrite(out_dir+'/tmp/argmin_depth.png', (255 * torch.clamp((argmin_depth[0,0] - depth_values[0,0]) / (depth_values[0,-1] - depth_values[0,0]), 0, 1)).detach().cpu().numpy().astype(np.uint8))
            #cv2.imwrite(out_dir+'/tmp/argmin_normal.png', (255 * np.clip(0.5*(argmin_normal[0].detach().cpu().numpy().transpose(1,2,0) + 1.0), 0, 1)).astype(np.uint8)[:,:,::-1])
            for i in range(warped_log_imgs.size(1)):
                cv2.imwrite(out_dir+'/tmp/warped_'+str(i)+'.png', np.clip(255 * (torch.expm1(warped_log_imgs[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                #cv2.imwrite(out_dir+'/tmp/warped_refined_'+str(i)+'.png', np.clip(255 * (torch.expm1(warped_log_imgs_[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/warped_blurred_'+str(i)+'.png', np.clip(255 * (torch.expm1(warped_log_imgs_blurred[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/rendered_'+str(i)+'.png', np.clip(255 * (torch.expm1(est_warped_log_imgs_naive[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/rendered_refined_'+str(i)+'.png', np.clip(255 * (torch.expm1(est_warped_log_imgs_refined[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/rendered_blurred_'+str(i)+'.png', np.clip(255 * (torch.expm1(est_warped_log_imgs_blurred[0,i]) / 1000).detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/observed_'+str(i)+'.png', np.clip(255 * imgs[0,i].detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/sphere_'+str(i)+'.png', np.clip(255 * est_rmaps[0,i].detach().cpu().numpy().transpose(1,2,0)**(1/2.2), 0, 255).astype(np.uint8)[:,:,::-1])
                cv2.imwrite(out_dir+'/tmp/occlusion_mask_'+str(i)+'.png', np.clip(255 * occlusion_masks[0,i,0].detach().cpu().numpy(), 0, 255).astype(np.uint8))

        scheduler.step()
        print('lr=', scheduler.get_last_lr())

        # save results
        loss_list.append(total_loss / (idx_batch+1))

        for i in range(3):
            plt.subplot(4,4,1+i)
            plot_hdr(imgs[:,i] / torch.clamp(torch.max(imgs), None, 1.0))
            if i == 0:
                plt.ylabel('Input')
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)

        plt.subplot(4,4,4)
        plot_hdr(illum_map)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,5)
        plot_hdr(recon_ref_img / torch.clamp(torch.max(imgs), None, 1.0))
        plt.ylabel('Recovered')
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,6)
        plot_hdr(est_rmaps[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,7)
        plot_normal_map(est_normal)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,8)
        plt.imshow(est_depth[0,0].detach().cpu().numpy(), vmin=depth_values[0,0].item(), vmax=depth_values[0,-1].item())
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,9)
        white_img = torch.ones_like(imgs[:,0])
        plot_hdr(white_img)
        plt.ylabel('GT')
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        if not (gt_rmaps is None):
            plt.subplot(4,4,10)
            plot_hdr(gt_rmaps[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)

        plt.subplot(4,4,11)
        plot_normal_map(gt_normal)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,12)
        plt.imshow(gt_depth[0,0].detach().cpu().numpy(), vmin=depth_values[0,0].item(), vmax=depth_values[0,-1].item())
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,13)
        plt.imshow(torch.mean(error_map.detach(), dim=1).cpu().numpy()[0], vmin=0, vmax=0.3)
        plt.ylabel('Error')
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        if not (gt_rmaps is None):
            est_log_rmap = est_log_rmaps[:,0]
            gt_log_rmap = torch.log1p(torch.clamp(1000 * gt_rmaps[:,0], 0, None))
            rmap_error = torch.mean(torch.abs(est_log_rmap - gt_log_rmap), dim=1, keepdim=True)
            plt.subplot(4,4,14)
            plt.imshow(torch.mean(rmap_error.detach(), dim=1).cpu().numpy()[0], vmin=0, vmax=0.3)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)

        plt.subplot(4,4,15)
        normal_cosine = torch.sum(est_normal * gt_normal, dim=1, keepdim=True)
        normal_error = torch.acos(torch.clamp(normal_cosine, -0.999, 0.999)) * masks[:,0]
        plt.imshow(normal_error[0,0].detach().cpu().numpy(), vmin=0.0, vmax=25.0/180.0*np.pi)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.subplot(4,4,16)
        depth_error = masks[:,0] * torch.abs(est_depth - gt_depth) / (depth_values[:,-1] - depth_values[:,0])
        plt.imshow(depth_error[0,0].detach().cpu().numpy(), vmin=0.0, vmax=0.1)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.box(False)

        plt.savefig(out_dir+'/'+str(idx_epoch).zfill(3)+'.png')
        plt.close()

        plt.subplot(2,2,1)
        plt.plot(loss_list)
        plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('image reconstruction loss')
        plt.subplot(2,2,2)
        plt.plot(dn_error_list)
        plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('depth-normal consistency error')
        plt.subplot(2,2,3)
        plt.plot(depth_error_list)
        plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('depth error')
        plt.subplot(2,2,4)
        plt.plot(normal_error_list)
        plt.grid()
        plt.xlabel('iteration')
        plt.ylabel('normal error')
        plt.savefig(out_dir+'/error.png')
        plt.close()

        brdf_result = {
            'embed_code': embed_code.detach().cpu(),
            'log_color': log_color.detach().cpu(),
        }
        torch.save(brdf_result, out_dir+'/brdf_code_'+str(idx_epoch).zfill(2)+'.pt')

        result = {
            'idx_epoch': idx_epoch,
            'embed_code': embed_code.detach().cpu(),
            'log_color': log_color.detach().cpu(),
            'brdf': brdf.detach().cpu(),
            'loss_list': loss_list,
            'dn_error_list': dn_error_list,
            'depth_error_list': depth_error_list,
            'normal_error_list': normal_error_list,
            'oprimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        torch.save(result, out_dir+'/result_tmp.pt')

    if not wo_sfs:
        torch.save(result, out_dir+'/result.pt')

        loss_list = result['loss_list']
        idx_epoch_best = 11 + np.argmin(loss_list[11:]) - 1 # ignore the first 11 iterations
        #idx_epoch_best = np.argmin(loss_list) - 1
        print('image reconstruction error is the lowest at epoch', idx_epoch_best)

        brdf_result_best = torch.load(out_dir+'/brdf_code_'+str(idx_epoch_best).zfill(2)+'.pt')
        with torch.no_grad():
            embed_code[:] = brdf_result_best['embed_code']
            log_color[:] = brdf_result_best['log_color']

    # save final results
    out_dir = out_dir+'/final_results'
    os.makedirs(out_dir, exist_ok=True)

    # save estimated BRDF
    with torch.no_grad():
        brdf = decode_brdf(ibrdf, embed_code, log_color)
    if not wo_sfs:
        SaveMERL(out_dir+'/brdf.binary', brdf[0])

        # visualize estimated BRDF as spheres lit by directional lights
        img, mask = visualize_merl_as_sheres(out_dir+'/brdf.binary')
        cv2.imwrite(out_dir+'/brdf.exr', img[:,:,::-1])

        img = np.clip(img**(1/2.2), 0.0, 1.0)
        img = img / np.clip(np.max(img), 1e-9,1)
        img = np.concatenate([img,mask[:,:,None]], axis=-1)

        img[:,:,:3] = img[:,:,:3][:,:,::-1]
        cv2.imwrite(out_dir+'/brdf.png', (255*img).astype(np.uint8))

        # visualize estimated BRDF as spheres lit by directional lights
        img, mask = visualize_merl_as_sheres(out_dir+'/brdf.binary', 45)
        cv2.imwrite(out_dir+'/brdf_45.exr', img[:,:,::-1])

        img = np.clip(img**(1/2.2), 0.0, 1.0)
        img = img / np.clip(np.max(img), 1e-9,1)
        img = np.concatenate([img,mask[:,:,None]], axis=-1)

        img[:,:,:3] = img[:,:,:3][:,:,::-1]
        cv2.imwrite(out_dir+'/brdf_45.png', (255*img).astype(np.uint8))

    # save estimated depths and normals
    renderer = Renderer(use_importance_sampling=True) # reset renderer
    with torch.no_grad():
        bar = tqdm(testloader)
        bar.set_description('Shape estimation (final)')
        list_est_depth = []
        list_est_normal = []
        for idx_view, minbatch in enumerate(bar):
            # load images, illumination map, and camera parameters
            imgs = minbatch['hdr_images'].to(device)
            masks = minbatch['masks'].to(device)
            illum_map = minbatch['illum_map'].to(device)
            intrinsics = minbatch['intrinsics'].to(device)
            extrinsics = minbatch['extrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            depth_ranges = minbatch['depth_ranges'].to(device)
            view_indices = minbatch['view_indices'].to(device)

            rot_matrices = extrinsics[:,:,:3,:3]

            illum_map = F.interpolate(illum_map, (256,512), mode='area')

            # load gt data
            gt_depth = None if not 'gt_depths' in minbatch else minbatch['gt_depths'].to(device)[:,0]
            gt_normal = None if not 'gt_normals' in minbatch else minbatch['gt_normals'].to(device)[:,0]
            gt_rmaps = None if not 'hdr_rmaps' in minbatch else minbatch['hdr_rmaps'].to(device)

            # make sure that the gt normal is normalized
            if not (gt_normal is None):
                l = torch.sqrt(torch.clamp(torch.sum(gt_normal**2, dim=1, keepdim=True), 1e-2, None))
                gt_normal = (gt_normal / l)

            # compute depth values
            depth_values = get_depth_values(
                torch.mean(depth_ranges[:,0], dim=1), 
                imgs.size()[3:5], 
                intrinsics[:,0], 
                numdepth=numdepth
            )

            # adjust depth_values to gt depth range
            est_depth_range = est_depth_range_list[idx_view].to(device)
            center_gt_depth = torch.exp(0.5 * (torch.log(est_depth_range[0]) + torch.log(est_depth_range[1])))
            center_depth_values = torch.exp(0.5 * (torch.log(depth_values[:,0]) + torch.log(depth_values[:,-1])))
            depth_values *= center_gt_depth / center_depth_values

            # render rmaps
            if wo_sfs:
                est_rmaps = torch.zeros((imgs.size(0),imgs.size(1),3,128,128), dtype=imgs.dtype, device=imgs.device)
            else:
                est_rmaps = render_sphere(
                    renderer, 
                    brdf.repeat(extrinsics.size(1),1,1,1,1), # [BS*N,3,90,90,180]
                    illum_map.repeat(extrinsics.size(1),1,1,1), # [BS*N,3,He,We]
                    extrinsics.view(-1,4,4), # [BS*N,4,4]
                    spp=spp_final
                ).view(extrinsics.size(0), extrinsics.size(1), 3, 128, 128)

            # make rmap_masks
            v,u = torch.meshgrid(torch.arange(est_rmaps.size(3)), torch.arange(est_rmaps.size(4)))
            x = 2 * (u + 0.5) / 128 - 1
            y = -(2 * (v + 0.5) / 128 - 1)
            z = torch.sqrt(torch.clamp(1-x**2-y**2,0,None))
            rmap_mask = (z > 0.0).float()[None,None].to(device)
            rmap_masks = rmap_mask[:,None,:,:,:].repeat(1,est_rmaps.size(1),1,1,1)

            # estimate shape as volumes
            out = mvsfsnet(imgs, est_rmaps.detach(), rot_matrices, proj_matrices, depth_values)
            est_depth = out['depth'][:,None] * masks[:,0]
            est_normal = out['normal'] * masks[:,0]
            depth_prob_volume = out['depth_prob_volume'] * masks[:,0,0,None]
            normal_volume = out['normal_volume'] * masks[:,0,:,None]

            normal_from_depth = depth_to_normal(est_depth[:,0], intrinsics[:,0]) * masks[:,0]

            list_est_depth.append(est_depth.cpu())
            list_est_normal.append(est_normal.cpu())

            # evaluate depth-normal consistency
            dn_error = dn_consistency_loss(depth_prob_volume, normal_volume, masks[:,0], intrinsics[:,0], depth_values)

            # compute normal / depth errors
            depth_error = compute_depth_error(est_depth, gt_depth, masks[:,0], depth_values)
            normal_error = compute_normal_error(est_normal, gt_normal, masks[:,0])
            num_pixels = torch.sum(masks[:,0])

            # save illum_map
            if idx_view == 0:
                save_hdr_as_ldr(out_dir+'/illumination.png', illum_map)

            # save per-view result
            per_view_out_dir = out_dir+'/view-'+str(idx_view+1).zfill(2)
            os.makedirs(per_view_out_dir, exist_ok=True)
            # reconstruction errors
            per_view_err_dict = {
                'depth_mae': depth_error.item(),
                'normal_mae_deg': np.degrees(normal_error.item()),
                'num_pixels': num_pixels.item(),
            }
            if not (bbox_diagonal is None):
                per_view_err_dict['depth_mae'] = depth_error.item() / bbox_diagonal
            with open(per_view_out_dir+'/errors.json', 'w') as f:
                json.dump(per_view_err_dict, f, ensure_ascii=True)

            def plot_hist(error_map, mask, range=[0,1]):
                plt.hist(error_map.reshape(-1)[mask.reshape(-1)>0], bins=100, range=range)
            np.savetxt(per_view_out_dir+'/depth_range.txt', np.array([depth_values[0,0].item(), depth_values[0,-1].item()]))
            np.save(per_view_out_dir+'/intrinsics.npy', intrinsics[0,0].cpu().numpy())
            np.save(per_view_out_dir+'/extrinsics.npy', extrinsics[0,0].cpu().numpy())
            for i in range(imgs.size(1)):
                save_hdr_as_ldr(per_view_out_dir+'/img_'+str(i)+'.png', imgs[:,i] / torch.clamp(torch.max(imgs), None, 1.0))
                #SaveHDR(per_view_out_dir+'/warped_img_'+str(i)+'.exr', warped_imgs[:,i])
                #save_hdr_as_ldr(per_view_out_dir+'/warped_img_'+str(i)+'.png', warped_imgs[:,i] / torch.clamp(torch.max(imgs), None, 1.0))
                #SaveHDR(per_view_out_dir+'/rendered_img_'+str(i)+'.exr', rendered_imgs[:,i])
                #save_hdr_as_ldr(per_view_out_dir+'/rendered_img_'+str(i)+'.png', rendered_imgs[:,i] / torch.clamp(torch.max(imgs), None, 1.0))
                #SaveHDR(per_view_out_dir+'/image_error_'+str(i)+'.exr', image_errors[:,i])
                #image_error_cm = cv2.applyColorMap((255*np.clip(image_errors[0,i].detach().cpu().numpy() / np.log(2),0,1)).astype(np.uint8), cv2.COLORMAP_JET)
                #cv2.imwrite(per_view_out_dir+'/image_error_'+str(i)+'.png', image_error_cm)
                #plt.figure(figsize=(12, 12))
                #plot_hist(image_errors[0,i].detach().cpu().numpy(), masks[0,0].detach().cpu().numpy(), range=[0,np.log(2)])
                #plt.xlabel('Log Error')
                #plt.ylabel('Number of Pixels')
                #plt.savefig(per_view_out_dir+'/image_error_hist_'+str(i)+'.png')
                #plt.close()
            cv2.imwrite(per_view_out_dir+'/ref_mask.png', (255 * masks[0,0,0]).detach().cpu().numpy().astype(np.uint8))

            #save_hdr_as_ldr(per_view_out_dir+'/recon_img.png', recon_ref_img / torch.clamp(torch.max(imgs), None, 1.0))
            def save_rmap(path, rmap):
                rmap_ldr = (torch.clamp(rmap[0], 0, 1)**(1/2.2) * 255).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]
                rmap_ldr = np.concatenate([rmap_ldr, (255*rmap_mask[0,0].detach().cpu().numpy().astype(np.uint8))[:,:,None]], axis=2)
                cv2.imwrite(path, rmap_ldr)

            save_rmap(per_view_out_dir+'/est_rmap.png', est_rmaps[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
            SaveHDR(per_view_out_dir+'/est_rmap.exr', est_rmaps[:,0])
            if not (gt_rmaps is None):
                save_rmap(per_view_out_dir+'/gt_rmap.png', gt_rmaps[:,0] / torch.clamp(torch.max(imgs), None, 1.0))
                SaveHDR(per_view_out_dir+'/gt_rmap.exr', gt_rmaps[:,0])
                rmap_error = torch.mean(torch.abs(torch.log1p(torch.clamp(1000 * est_rmaps[:,0], 0, None)) - torch.log1p(torch.clamp(1000 * gt_rmaps[:,0], 0, None))), dim=1)[0].detach().cpu().numpy()
                rmap_error_cm = cv2.applyColorMap((255*np.clip(rmap_error / np.log(2),0,1)).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(per_view_out_dir+'/rmap_error.exr', rmap_error)
                cv2.imwrite(per_view_out_dir+'/rmap_error.png', np.concatenate([rmap_error_cm, (255*rmap_mask[0,0].detach().cpu().numpy().astype(np.uint8))[:,:,None]], axis=2))
                plt.figure(figsize=(12, 12))
                plot_hist(rmap_error, rmap_mask[0,0].detach().cpu().numpy(), range=[0,np.log(2)])
                plt.xlabel('Log Error')
                plt.ylabel('Number of Pixels')
                plt.savefig(per_view_out_dir+'/rmap_error_hist.png')
                plt.close()

            np.save(per_view_out_dir+'/est_normal.npy', (est_normal * masks[:,0])[0].cpu().numpy().transpose(1,2,0))
            save_normal_map(per_view_out_dir+'/est_normal.png', est_normal * masks[:,0])
            save_normal_map(per_view_out_dir+'/est_depth_derivative.png', normal_from_depth * masks[:,0])
            save_normal_map(per_view_out_dir+'/gt_normal.png', gt_normal * masks[:,0])
            normal_error = (torch.acos(torch.clamp(torch.sum(est_normal * gt_normal, dim=1),-1,1)) * masks[:,0,0])[0].detach().cpu().numpy()
            normal_error_gray = (255 * np.clip(normal_error / np.radians(30), 0, 1)).astype(np.uint8)
            normal_error_cmap = cv2.applyColorMap(normal_error_gray, cv2.COLORMAP_JET)
            cv2.imwrite(per_view_out_dir+'/normal_error.exr', normal_error)
            cv2.imwrite(per_view_out_dir+'/normal_error.png', normal_error_cmap)
            plot_hist(np.degrees(normal_error), masks[0,0].detach().cpu().numpy(), range=[0,30])
            plt.xlabel('Normal Error[Deg]')
            plt.ylabel('Number of Pixels')
            plt.savefig(per_view_out_dir+'/normal_error_hist.png')
            plt.close()
            dn_error = (torch.acos(torch.clamp(torch.sum(est_normal * normal_from_depth, dim=1),-1,1)) * masks[:,0,0])[0].detach().cpu().numpy()
            dn_error_gray = (255 * np.clip(dn_error / np.radians(30), 0, 1)).astype(np.uint8)
            dn_error_cmap = cv2.applyColorMap(dn_error_gray, cv2.COLORMAP_JET)
            cv2.imwrite(per_view_out_dir+'/depth_normal_error.exr', dn_error)
            cv2.imwrite(per_view_out_dir+'/depth_normal_error.png', dn_error_cmap)
            plot_hist(np.degrees(dn_error), masks[0,0].detach().cpu().numpy(), range=[0,30])
            plt.xlabel('Normal Error[Deg]')
            plt.ylabel('Number of Pixels')
            plt.savefig(per_view_out_dir+'/depth_normal_error_hist.png')
            plt.close()

            np.save(per_view_out_dir+'/est_depth.npy', (est_depth * masks[:,0])[0,0].cpu().numpy())
            est_depth_gray = (255 * torch.clamp(((est_depth * masks[:,0])[0,0] - depth_values[0,0]) / (depth_values[0,-1] - depth_values[0,0]), 0, 1)).detach().cpu().numpy().astype(np.uint8)
            est_depth_cmap = cv2.applyColorMap(est_depth_gray, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(per_view_out_dir+'/est_depth.png', est_depth_cmap)
            gt_depth_gray = (255 * torch.clamp(((gt_depth * masks[:,0])[0,0] - depth_values[0,0]) / (depth_values[0,-1] - depth_values[0,0]), 0, 1)).detach().cpu().numpy().astype(np.uint8)
            gt_depth_cmap = cv2.applyColorMap(gt_depth_gray, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(per_view_out_dir+'/gt_depth.png', gt_depth_cmap)
            if not (bbox_diagonal is None):
                depth_error = (torch.abs(est_depth - gt_depth) * masks[:,0] / bbox_diagonal)[0,0].detach().cpu().numpy()
                depth_error_cm = cv2.applyColorMap((255*np.clip(depth_error / 0.03,0,1)).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(per_view_out_dir+'/depth_error.exr', depth_error)
                cv2.imwrite(per_view_out_dir+'/depth_error.png', depth_error_cm)
                plt.figure(figsize=(12, 12))
                plot_hist(100*depth_error, masks[0,0].detach().cpu().numpy(), range=[0,3])
                plt.xlabel('Depth Error[%]')
                plt.ylabel('Number of Pixels')
                plt.savefig(per_view_out_dir+'/depth_error_hist.png')
                plt.close()

            with open(per_view_out_dir+'/views.txt', 'w') as f:
                s = ' '.join([str(val.item()) for val in view_indices[0]])
                f.write(s)


        bar = tqdm(testloader)
        bar.set_description('Occlusion evaluation (final)')
        for idx_view, minbatch in enumerate(bar):
            ref_mask = minbatch['masks'].to(device)[:,0]
            extrinsics = minbatch['extrinsics'].to(device)
            proj_matrices = minbatch['proj_matrices'].to(device)
            view_indices = minbatch['view_indices'].to(device)

            rot_matrices = extrinsics[:,:,:3,:3]

            if len(test_dataset) == 1:
                vis_mask = ref_mask
            else:
                # filter results by leveraging approximated visibility
                est_depth = list_est_depth[idx_view].to(device)
                est_depths = torch.stack([list_est_depth[i].to(device) for i in view_indices[0]], dim=1)
                est_normals = torch.stack([list_est_normal[i].to(device) for i in view_indices[0]], dim=1)

                est_depths_warped, est_depths_sampled = warp_depth_maps(est_depths, proj_matrices, est_depth)
                est_normals_warped = warp_normal_maps(est_normals, rot_matrices, proj_matrices, est_depth)

                est_normals_cosine = torch.sum(est_normals[:,0:1] * est_normals_warped, dim=2, keepdim=True)
                est_normals_cosine = torch.clamp(est_normals_cosine,0,1) * (est_normals_warped[:,:,2:3] > 0.0).float()

                vis_mask_n = (torch.sum((est_normals_cosine > np.cos(threshold_final_normal_error)).float(), dim=1) >= 3).float()
                vis_mask = ref_mask * vis_mask_n

                if not (bbox_diagonal is None):
                    est_depths_error = torch.abs(est_depths_warped - est_depths_sampled) / bbox_diagonal
                    vis_mask_d = (torch.sum((est_depths_error < threshold_final_depth_error).float(), dim=1) >= 3.0).float()
                    vis_mask *= vis_mask_d

            if False:
                for i in range(imgs.size(1)):
                    plt.subplot(7,imgs.size(1),1+i)
                    est_depth_range = est_depth_range_list[view_indices[0][i]]
                    plt.imshow(est_depths[0,i,0].detach().cpu().numpy(), vmin=est_depth_range[0].item(),vmax=est_depth_range[1].item())
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('est_depth')
                    plt.subplot(7,imgs.size(1),1+imgs.size(1)+i)
                    plot_normal_map(est_normals[:,i])
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('est_normal')
                    plt.subplot(7,imgs.size(1),1+2*imgs.size(1)+i)
                    est_depth_range = est_depth_range_list[view_indices[0][0]]
                    plt.imshow(est_depths_warped[0,i,0].detach().cpu().numpy(), vmin=est_depth_range[0].item(),vmax=est_depth_range[1].item())
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('est_depth_warped')
                    plt.subplot(7,imgs.size(1),1+3*imgs.size(1)+i)
                    plot_normal_map(est_normals_warped[:,i])
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('est_normal_warped')
                    plt.subplot(7,imgs.size(1),1+4*imgs.size(1)+i)
                    plt.imshow(est_depths_error[0,i,0].cpu(), vmin=0, vmax=0.1)
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('depth_error')
                    plt.subplot(7,imgs.size(1),1+5*imgs.size(1)+i)
                    plt.imshow(np.degrees(np.arccos(est_normals_cosine[0,i,0].cpu())), vmin=0, vmax=30)
                    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                    plt.tick_params(bottom=False,left=False,right=False,top=False)
                    plt.box(False)
                    if i == 0:
                        plt.ylabel('normal_error')
                plt.subplot(7,imgs.size(1),1+6*imgs.size(1))
                plt.imshow(vis_mask_d[0,0].cpu())
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.ylabel('depth_mask')
                plt.subplot(7,imgs.size(1),1+6*imgs.size(1)+1)
                plt.imshow(vis_mask_n[0,0].cpu())
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.ylabel('normal_mask')
                plt.subplot(7,imgs.size(1),1+6*imgs.size(1)+2)
                plt.imshow(vis_mask[0,0].cpu())
                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
                plt.tick_params(bottom=False,left=False,right=False,top=False)
                plt.box(False)
                plt.ylabel('vis_mask')
                plt.show()

            per_view_out_dir = out_dir+'/view-'+str(idx_view+1).zfill(2)
            cv2.imwrite(per_view_out_dir+'/vis_mask.png', (255 * vis_mask[0,0]).detach().cpu().numpy().astype(np.uint8))