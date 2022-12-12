import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_hdr(hdr_img, idx_batch=0):
    if hdr_img.ndim == 4:
        hdr_img = hdr_img[idx_batch]
    img = torch.clamp(hdr_img**(1/2.2), 0, 1).detach().cpu().numpy().transpose((1,2,0))
    plt.imshow(img)

def plot_normal_map(normal_map, idx_batch=0):
    if normal_map.ndim == 4:
        normal_map = normal_map[idx_batch]
    normal_map = (0.5*(normal_map + 1)).detach().cpu().numpy().transpose((1,2,0))
    plt.imshow(normal_map)


def plot_normal_prob(prob_volume, idx_batch, u, v, gt_normal=None):
    log_prob_dist = torch.log(prob_volume[idx_batch,:,:,v,u]+1e-6)
    log_prob_dist = log_prob_dist.detach().cpu()
    log_prob_dist = log_prob_dist - torch.max(log_prob_dist)

    # mask meaningless pixels
    BS,Hn,Wn,H,W = prob_volume.size()
    v_grid,u_grid = torch.meshgrid(torch.arange(Hn),torch.arange(Wn))
    ny_grid = 2 * (v_grid + 0.5).float() / Hn - 1
    nx_grid = 2 * (u_grid + 0.5).float() / Wn - 1
    mask = (nx_grid**2 + ny_grid**2) < 1.0

    log_prob_dist[mask != True] = np.inf
    
    plt.imshow(log_prob_dist.numpy(), vmin=-9, vmax=0)

    if gt_normal != None:
        nx, ny, nz = gt_normal[idx_batch,:,v,u].detach().cpu().numpy()
        idx_nx = log_prob_dist.size(1) * 0.5*(nx + 1) - 0.5
        idx_ny = log_prob_dist.size(0) * 0.5*(-ny + 1) - 0.5
        plt.scatter([idx_nx], [idx_ny], marker='o', s=200,  facecolor='None', edgecolors='red', linewidths=4)

def save_hdr_as_ldr(dst, hdr, gamma=2.2, clip_value=1.0, idx_batch=0):
    ldr = torch.clamp(hdr.detach() / clip_value, 0, 1)**(1/gamma)
    ldr = ldr[idx_batch].detach().cpu().numpy().transpose(1,2,0)
    cv2.imwrite(dst, (ldr[:,:,::-1]*255).astype(np.uint8))

def save_normal_map(dst, normal_map, idx_batch=0):
    if normal_map.ndim == 4:
        normal_map = normal_map[idx_batch]
    normal_map = (0.5*(normal_map.detach() + 1)).cpu().numpy().transpose((1,2,0))
    cv2.imwrite(dst, (normal_map[:,:,::-1]*255).astype(np.uint8))
        
def sample_normal_prob(prob_volume, normal_map):
    BS,Hn,Wn,H,W = prob_volume.size()
    pv = prob_volume.view(BS,Hn*Wn,H*W).transpose(1,2).contiguous().view(BS*H*W,1,Hn,Wn)
    grid = normal_map[:,:2].contiguous().view(BS,2,H*W).transpose(1,2).contiguous().view(BS*H*W,1,1,2)
    grid[:,:,:,1] = grid[:,:,:,1] * (-1)
    sampled = F.grid_sample(pv, grid, mode='bilinear', padding_mode='border', align_corners=False) # [BS*H*W,1,1,1]
    sampled = sampled[:,0,0,0].contiguous().view(BS,1,H,W)
    return sampled

def sample_reflectance_map(reflectance_map, normal_map):
    grid = torch.stack([normal_map[:,0],-normal_map[:,1]], dim=-1)
    mask = (normal_map[:,2:3] > 0.0).float()
    sampled = F.grid_sample(reflectance_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    sampled = sampled * mask
    return sampled

def sample_reflectance_map_prob(reflectance_map, prob_volume):
    BS,Hn,Wn,H,W = prob_volume.size()
    rmap = F.interpolate(reflectance_map, (Hn,Wn), mode='area', align_corners=None)
    rmap2 = F.interpolate(reflectance_map**2, (Hn,Wn), mode='area', align_corners=None)

    x = torch.sum(rmap[:,:,:,:,None,None] * prob_volume[:,None], dim=(2,3))
    x2 = torch.sum(rmap2[:,:,:,:,None,None] * prob_volume[:,None], dim=(2,3))

    mu = x
    var = x2 - x**2

    return mu, var

def eval_negative_log_likelihood(x, mu, var, eps=1e-9):
    var = torch.clamp(var,0,None) + eps
    return 0.5 * torch.log(2*np.pi*var) + (x-mu)**2/(2*var)

def compute_mean_normal(prob_volume):
    BS,Hn,Wn,H,W = prob_volume.size()
    device = prob_volume.device
    grid_ny, grid_nx = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    grid_ny = -(2 * (grid_ny.float().to(device) + 0.5) / Hn - 1.0)
    grid_nx = 2 * (grid_nx.float().to(device) + 0.5) / Wn - 1.0
    grid_nz = torch.sqrt(torch.clamp(1 - grid_nx**2 - grid_ny**2, 0, 1))
    grid = torch.stack([grid_nx, grid_ny, grid_nz], dim=0) # 3,Hn,Wn

    mean_normal = torch.sum(prob_volume[:,None,:,:,:,:] * grid[None,:,:,:,None,None], dim=(2,3))
    mean_normal_norm = torch.sqrt(torch.sum(mean_normal**2, dim=1, keepdim=True))
    mean_normal = mean_normal / torch.clamp(mean_normal_norm, 1e-4, None)
    return mean_normal, mean_normal_norm

def compute_argmax_normal(prob_volume):
    BS,Hn,Wn,H,W = prob_volume.size()
    device = prob_volume.device
    grid_ny, grid_nx = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    grid_ny = -(2 * (grid_ny.float().to(device) + 0.5) / Hn - 1.0)
    grid_nx = 2 * (grid_nx.float().to(device) + 0.5) / Wn - 1.0
    grid_nz = torch.sqrt(torch.clamp(1 - grid_nx**2 - grid_ny**2, 0, 1))
    grid = torch.stack([grid_nx, grid_ny, grid_nz], dim=0) # 3,Hn,Wn

    idx_argmax_u = torch.max(torch.max(prob_volume.detach(), dim=1)[0], dim=1)[1] # [BS,W,H]
    idx_argmax_v = torch.max(torch.max(prob_volume.detach(), dim=2)[0], dim=1)[1] # [BS,W,H]

    argmax_nx = 2 * (idx_argmax_u + 0.5) / Wn - 1.0
    argmax_ny = -(2 * (idx_argmax_v + 0.5) / Hn - 1.0)
    argmax_nz = torch.sqrt(torch.clamp(1 - argmax_nx**2 - argmax_ny**2, 0, 1))

    argmax_normal = torch.stack([argmax_nx, argmax_ny, argmax_nz], dim=1)
    return argmax_normal

def compute_soft_argmax_normal(prob_volume):
    BS,Hn,Wn,H,W = prob_volume.size()
    device = prob_volume.device
    grid_ny, grid_nx = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    grid_ny = -(2 * (grid_ny.float().to(device) + 0.5) / Hn - 1.0)
    grid_nx = 2 * (grid_nx.float().to(device) + 0.5) / Wn - 1.0
    grid_nz = torch.sqrt(torch.clamp(1 - grid_nx**2 - grid_ny**2, 0, 1))
    grid = torch.stack([grid_nx, grid_ny, grid_nz], dim=0) # 3,Hn,Wn

    idx_argmax_u = torch.max(torch.max(prob_volume.detach(), dim=1)[0], dim=1)[1] # [BS,W,H]
    idx_argmax_v = torch.max(torch.max(prob_volume.detach(), dim=2)[0], dim=1)[1] # [BS,W,H]

    argmax_nx = 2 * (idx_argmax_u + 0.5) / Wn - 1.0
    argmax_ny = -(2 * (idx_argmax_v + 0.5) / Hn - 1.0)

    d2 = (grid[0][None,:,:,None,None] - argmax_nx[:,None,None,:,:])**2 + (grid[1][None,:,:,None,None] - argmax_ny[:,None,None,:,:])**2
    d2 = d2 / (2/Hn)
    mask = torch.exp(-5*d2)
    mask = mask.float()

    #plt.imshow(mask[0,:,:,64,64].detach().cpu().numpy())
    #plt.show()

    prob_volume = prob_volume * mask

    mean_normal = torch.sum(prob_volume[:,None,:,:,:,:] * grid[None,:,:,:,None,None], dim=(2,3))
    mean_normal_norm = torch.sqrt(torch.sum(mean_normal**2, dim=1, keepdim=True))
    mean_normal = mean_normal / torch.clamp(mean_normal_norm, 1e-4, None)
    return mean_normal

# img: BS,C,H,W
# extrinsic: BS,4,4
def illum_map_to_mirror_sphere(illum_map, extrinsic, out_size=(128,128)):
    device = illum_map.device
    dtype = illum_map.dtype
    Wn,Hn = out_size
    v,u = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    nx = 2 * (u.to(device) + 0.5) / Wn - 1
    ny = 2 * (v.to(device) + 0.5) / Hn - 1
    nz = -torch.sqrt(torch.clamp(1-nx**2-ny**2,0,None))
    grid_n = torch.stack([nx,ny,nz], dim=2) # [H,W,3]
    v = torch.tensor([0.0,0.0,-1.0], dtype=dtype).to(device)[None,None,:]
    grid_l = -v + 2 * torch.sum(v*grid_n, dim=2, keepdim=True) * grid_n
    mask = (nz < 0.0).float()
    
    grid_l = torch.matmul(extrinsic[:,None,None,:3,:3].transpose(3,4), grid_l[None,:,:,:,None])[:,:,:,:,0]
    grid_theta = torch.acos(torch.clamp(grid_l[:,:,:,1], -1, 1))
    grid_phi = torch.atan2(grid_l[:,:,:,2], grid_l[:,:,:,0])
    grid_phi[grid_phi < 0.0] += 2.0 * np.pi
    u = grid_phi/np.pi - 1.0
    v = 2 * grid_theta / np.pi - 1.0
    grid = torch.stack([u, v], dim=-1)
    out = F.grid_sample(illum_map, grid, mode='bilinear', padding_mode='border')
    return out * mask

# hdr:  [...,3,H,W]
# mask: [...,3,H,W]
def preprocess_hdr(hdr,mask):
    hdr[torch.isnan(hdr)] = 0
    hdr[torch.isinf(hdr)] = 1e3
    mask[torch.isnan(mask)] = 0
    mask[torch.isinf(mask)] = 1   
    mask[mask<0.5] = 0.0
    mask[mask>=0.5] = 1.0 
    hdr = torch.clamp(hdr, 0, 1e3)
    log_hdr = torch.log(hdr+1e-3)*mask
    return torch.cat([log_hdr, mask], dim=-3)