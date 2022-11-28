import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from core.dataset import PreProcessedDataset
from core.sfs_models import SfSNet
from core.sfs_criterion import NormalLogLikelihoodLoss, NormalAccuracy
from core.sfs_utils import *

import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob

from tqdm import tqdm

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS = 2
weight_dir = './weights/sfsnet'
os.makedirs(weight_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default=os.environ['HOME']+'/data/tmp/sfs')
args = parser.parse_args()

dataset = PreProcessedDataset(args.dataset_dir)

list_split = np.arange(len(dataset))
train_subset_indices =  list_split[:int(0.8*len(list_split))]
train_dataset = Subset(dataset, train_subset_indices)
trainloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)

val_subset_indices =  list_split[int(0.8*len(list_split)):int(0.9*len(list_split))]
val_dataset = Subset(dataset, val_subset_indices)
valloader = DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)

# load model
sfsnet = SfSNet()
sfsnet = nn.DataParallel(sfsnet)
sfsnet.to(device)

criterion = NormalLogLikelihoodLoss()
criterion_acc = NormalAccuracy()

# optimizer
optimizer = torch.optim.Adam(sfsnet.parameters())

list_ckpt = sorted(glob.glob(weight_dir+'/???.ckpt'))
idx_itr_ofs = 0
if len(list_ckpt) > 0:
    path = list_ckpt[-1]
    checkpoint = torch.load(path)
    print('existing checkpoint '+path+' loaded')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sfsnet.module.load_state_dict(checkpoint['sfsnet_state_dict'])
    idx_itr_ofs = len(list_ckpt)

for idx_itr in range(idx_itr_ofs, 40):
    # train
    bar = tqdm(trainloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (train)')
    sfsnet.train()
    total_loss = 0.0
    total_acc = 0.0
    for idx_minbatch, minbatch in enumerate(bar):
        img = minbatch['img'].to(device)
        rmap = minbatch['rmap'].to(device)
        mask = minbatch['mask'].to(device)
        #rmap_diffuse = minbatch['rmap_diffuse'].to(device)
        #rmap_mirror = minbatch['rmap_mirror'].to(device)
        gt_normal = minbatch['normal'].to(device)

        results = sfsnet(img, rmap)

        multiscale_loss = []
        for result in results:
            loss_ = criterion(result['hypotheses_probability'], result['grid_normal'], result['patch_size'], gt_normal, mask)
            acc = criterion_acc(result['grid_normal'], result['patch_size'], gt_normal, mask)
            multiscale_loss.append(loss_)
        multiscale_loss = torch.stack(multiscale_loss)
        loss = torch.mean(multiscale_loss)

        # backward
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()        

        # update bar postfix
        total_loss += loss.item()
        total_acc += acc.item()
        bar.set_postfix(
            loss=loss.item(), 
            acc=acc.item(), 
            mean_loss=total_loss/(idx_minbatch+1),
            mean_acc=total_acc/(idx_minbatch+1),
        )
    train_loss = total_loss/(idx_minbatch+1)
    train_acc = total_acc/(idx_minbatch+1)

    # val
    bar = tqdm(valloader)
    bar.set_description('Epoch '+str(idx_itr+1).zfill(2)+' (val) ')
    sfsnet.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for idx_minbatch, minbatch in enumerate(bar):
            img = minbatch['img'].to(device)
            rmap = minbatch['rmap'].to(device)
            mask = minbatch['mask'].to(device)
            #rmap_diffuse = minbatch['rmap_diffuse'].to(device)
            #rmap_mirror = minbatch['rmap_mirror'].to(device)
            gt_normal = minbatch['normal'].to(device)

            results = sfsnet(img, rmap)

            multiscale_loss = []
            for result in results:
                loss_ = criterion(result['hypotheses_probability'], result['grid_normal'], result['patch_size'], gt_normal, mask)
                acc = criterion_acc(result['grid_normal'], result['patch_size'], gt_normal, mask)
                multiscale_loss.append(loss_)
            multiscale_loss = torch.stack(multiscale_loss)
            loss = torch.mean(multiscale_loss)

            # update bar postfix
            total_loss += loss.item()
            total_acc += acc.item()
            bar.set_postfix(
                loss=loss.item(), 
                acc=acc.item(), 
                mean_loss=total_loss/(idx_minbatch+1),
                mean_acc=total_acc/(idx_minbatch+1),
            )
    val_loss = total_loss/(idx_minbatch+1)
    val_acc = total_acc/(idx_minbatch+1)

    # save weights
    torch.save({
            'sfsnet_state_dict': sfsnet.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
    }, weight_dir+'/'+str(idx_itr).zfill(3)+'.ckpt') 