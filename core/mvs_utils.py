import torch
import torch.nn as nn
import torch.nn.functional as F

import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

import numpy as np

from .mvs_criterion import DepthNormalConsistencyLoss
from .module import homo_warping

# prob_volume: [BS,Hn,Wn,H,W]
# ref_rot: [BS,3,3]
# src_rot: [BS,3,3]
def rotate_rmap(rmap, ref_rot, src_rot):
    BS,C,Hn,Wn = rmap.size()
    device = rmap.device
    grid_ny, grid_nx = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    grid_ny = -(2 * (grid_ny.float().to(device) + 0.5) / Hn - 1.0)
    grid_nx = 2 * (grid_nx.float().to(device) + 0.5) / Wn - 1.0
    grid_nz = torch.sqrt(torch.clamp(1 - grid_nx**2 - grid_ny**2, 0, 1))
    normal = torch.stack([grid_nx, -grid_ny, -grid_nz], dim=2) # Hn,Wn,3    

    # compute transformation
    if (BS == 2) and (ref_rot.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_ref_rot = torch.stack([torch.inverse(m) for m in ref_rot], dim=0)
        rot = torch.matmul(src_rot, inv_ref_rot)
    else:
        rot = torch.matmul(src_rot, torch.inverse(ref_rot))
    normal = torch.matmul(rot[:,None,None,:,:], normal[None,:,:,:,None])[:,:,:,:,0]
    grid_nx1 = normal[:,:,:,0]
    grid_ny1 = -normal[:,:,:,1]
    grid_nz1 = torch.clamp(-normal[:,:,:,2],0,1)
    grid = torch.stack([grid_nx1, -grid_ny1], dim=3) # BS,Hn,Wn,2   

    # resampling
    rotated_rmap = F.grid_sample(rmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    rotated_rmap = rotated_rmap * (grid_nz1 > 0.0).float()[:,None,:,:]

    return rotated_rmap

# prob_volume: [BS,Hn,Wn,H,W]
# ref_rot: [BS,3,3]
# src_rot: [BS,3,3]
def rotate_prob_volume(prob_volume, ref_rot, src_rot):
    BS,Hn,Wn,H,W = prob_volume.size()
    device = prob_volume.device
    grid_ny, grid_nx = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
    grid_ny = -(2 * (grid_ny.float().to(device) + 0.5) / Hn - 1.0)
    grid_nx = 2 * (grid_nx.float().to(device) + 0.5) / Wn - 1.0
    grid_nz = torch.sqrt(torch.clamp(1 - grid_nx**2 - grid_ny**2, 0, 1))
    normal = torch.stack([grid_nx, -grid_ny, -grid_nz], dim=2) # Hn,Wn,3    

    # compute transformation
    if (BS == 2) and (ref_rot.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_ref_rot = torch.stack([torch.inverse(m) for m in ref_rot], dim=0)
        rot = torch.matmul(src_rot, inv_ref_rot)
    else:
        rot = torch.matmul(src_rot, torch.inverse(ref_rot))
    normal = torch.matmul(rot[:,None,None,:,:], normal[None,:,:,:,None])[:,:,:,:,0]
    grid_nx1 = normal[:,:,:,0]
    grid_ny1 = -normal[:,:,:,1]
    grid_nz1 = torch.clamp(-normal[:,:,:,2],0,1)
    grid = torch.stack([grid_nx1, -grid_ny1], dim=3) # BS,Hn,Wn,2   

    # resampling
    prob_volume_ = prob_volume.view(BS,Hn*Wn,H*W).transpose(1,2)
    prob_volume_ = prob_volume_.contiguous().view(BS,H*W,Hn,Wn)
    prob_volume_ = F.grid_sample(prob_volume_, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    prob_volume_ = prob_volume_.view(BS,H*W,Hn*Wn).transpose(1,2)
    prob_volume = prob_volume_.contiguous().view(BS,Hn,Wn,H,W)

    # reweighting & masking
    w = grid_nz1 / (grid_nz[None,:,:]+1e-6) * (grid_nz[None,:,:] > 0.0).float()
    prob_volume = prob_volume * w[:,:,:,None,None]

    return prob_volume

def warp_normal_maps(normal_maps, rot_matrices, proj_matrices, ref_depth_map):
    device = normal_maps.device
    # rotation
    n_ = normal_maps * torch.tensor([1.0,-1.0,-1.0], device=device)[None,None,:,None,None]
    rot_ = rot_matrices[:,0:1] @ torch.inverse(rot_matrices)
    n_ = torch.sum(rot_[:,:,:,:,None,None] * n_[:,:,None,:,:,:], dim=3)
    normal_maps_rotated = n_ * torch.tensor([1.0,-1.0,-1.0], device=device)[None,None,:,None,None]

    # warping
    warped_normal_maps = sample_from_src_views(
        ref_depth_map,
        normal_maps_rotated, 
        proj_matrices
    )[0] # [BS,N,3,H,W]

    l = torch.sqrt(torch.clamp(torch.sum(warped_normal_maps**2, dim=2, keepdim=True), 1e-2, None))
    warped_normal_maps = (warped_normal_maps / l)

    return warped_normal_maps

def warp_depth_maps(depth_maps, proj_matrices, ref_depth_map):
    d_sampled, proj_coords = sample_from_src_views(
        ref_depth_map,
        depth_maps, 
        proj_matrices
    ) # [BS,N,1,H,W],[BS,N,H,W,3]
    d_warped = proj_coords[...,2][:,:,None]
    return d_warped, d_sampled

# depth_center: (BS)
# imsize: (2)
# intrinsics: (BS,3,3)
# ret: (BS,D)
def get_depth_values(depth_center, imsize, intrinsic, numdepth=192):
    w0 = min(imsize)
    f = intrinsic[:,0,0]
    a = 1.5 / numdepth
    #print('depth0: ', depth0)
    #print('depth_ranges', depth_ranges[:,0])
    #print('focal_length: ', f)
    depth_values_exp = torch.arange(numdepth, device=intrinsic.device)[None] - (numdepth//2)
    depth_values = (f / (f - a*w0))[:,None]**depth_values_exp * depth_center[:,None]
    #print(depth_values) 
    return depth_values

# depth: BS,H,W
# intrinsic: BS,3,3
def depth_to_normal(depth, intrinsic):
    BS,H,W = depth.size()
    # compute transformation
    if (BS == 2) and (intrinsic.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_intrinsic = torch.stack([torch.inverse(m) for m in intrinsic], dim=0)
    else:
        inv_intrinsic = torch.inverse(intrinsic)
    v,u = torch.meshgrid(torch.arange(H), torch.arange(W))
    ones = torch.ones_like(v)
    m = torch.stack([u,v,ones], dim=2).float().to(depth.device) # [H,W,3]
    p0 = torch.matmul(inv_intrinsic[:,None,None,:,:], m[None,:,:,:,None])[:,:,:,:,0] # [BS,H,W,3]
    p  = depth[:,:,:,None] * p0 # [BS,H,W,3]
    def diff(x, dim):
        res = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        x = x.transpose(0,dim)
        res = res.transpose(0,dim)
        res[1:-1] = 0.5*(x[2:] - x[:-2])
        return res.transpose(0,dim)
    dpdv = diff(p,1)
    dpdu = diff(p,2)
    normal = -torch.cross(dpdu,dpdv, dim=3) # [BS,H,W,3]

    # to (right,up,-lookat)
    normal[:,:,:,1:] = -normal[:,:,:,1:]

    normal = torch.stack(torch.unbind(normal, dim=3), dim=1) # [BS,3,H,W]

    # normalize
    norm = torch.sqrt(torch.sum(normal.detach()**2, dim=1, keepdim=True)+1e-20)
    normal = normal / norm

    # masking using depth
    mask = (depth.detach() > 0.0).float()[:,None]
    mask[:,:,1:-1] *= mask[:,:,0:-2] * mask[:,:,2:]
    mask[:,:,:,1:-1] *= mask[:,:,:,0:-2] * mask[:,:,:,2:]
    normal = normal * mask

    return normal

def sample_from_src_views(ref_depth, features, proj_matrices):
    BS,N,C,H,W = features.size()

    v,u = torch.meshgrid(torch.arange(H),torch.arange(W))
    v = v.float().to(ref_depth.device)
    u = u.float().to(ref_depth.device)
    ones = torch.ones_like(v).float().to(ref_depth.device)
    m = torch.stack([u,v,ones], dim=2) # [H,W,3]
    m = m[None]*ref_depth[:,0,:,:,None] #[BS,H,W,3]
    _m_ref = torch.cat([m, ones[None,:,:,None].repeat(BS,1,1,1)], dim=3) # [BS,H,W,4]

    ref_proj = proj_matrices[:,0]  
    if (BS == 2) and (ref_proj.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_ref_proj = torch.stack([torch.inverse(m) for m in ref_proj], dim=0)
    else:
        inv_ref_proj = torch.inverse(ref_proj)   
        
    src_features = torch.unbind(features[:,1:], dim=1)
    src_projs = torch.unbind(proj_matrices[:,1:], dim=1)
    sampled_features = [features[:,0]]
    ref_proj_coord = torch.stack([u,v], dim=2)[None].repeat(BS,1,1,1) # [BS,H,W,2]
    ref_proj_coord = torch.cat([ref_proj_coord, ref_depth[:,0,:,:,None]], dim=-1) # [BS,H,W,2]
    proj_coords = [ref_proj_coord]
    for (src_fea, src_proj) in zip(src_features, src_projs):
        proj = torch.matmul(src_proj, inv_ref_proj) # [BS,4,4]
        _m_src = torch.matmul(proj[:,None,None,:,:], _m_ref[:,:,:,:,None])[:,:,:,:,0] # [BS,H,W,4]
        m_src = _m_src[:,:,:,:3] / _m_src[:,:,:,3:4]
        d_src = m_src[:,:,:,2]
        u_src = m_src[:,:,:,0] / d_src
        v_src = m_src[:,:,:,1] / d_src
        grid = 2 * torch.stack([u_src/W, v_src/H], dim=3) - 1 # [BS,H,W,2]
        fea_sampled = F.grid_sample(src_fea, grid, mode='bilinear', padding_mode='border', align_corners=False)
        sampled_features.append(fea_sampled)
        proj_coords.append(torch.stack([u_src,v_src,d_src], dim=3))
    sampled_features = torch.stack(sampled_features, dim=1)
    proj_coords = torch.stack(proj_coords, dim=1) # BS,N,H,W,3
    return sampled_features, proj_coords

# rmap: BS,C,Hn,Wn
# normal_volume: BS,3,D,H,W
# ret: BS,C,D,H,W
def construct_radiance_volume(rmap, normal_volume):
    BS,Cn,D,H,W = normal_volume.size()
    u = normal_volume[:,0]
    v = -normal_volume[:,1]
    grid = torch.stack([u,v], dim=4).view(BS,D*H,W,2)
    sampled_radiance = F.grid_sample(rmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return sampled_radiance.view(BS,-1,D,H,W)

# rmaps: BS,N,C,Hn,Wn
# rot_matrices: BS,N,3,3
# normal_volume: BS,3,D,H,W
# ret: BS,N,C,D,H,W
def construct_reflectance_feature_volumes(rmaps, rot_matrices, normal_volume):
    BS,Cn,D,H,W = normal_volume.size()

    ref_rmap = rmaps[:,0]
    ref_rot = rot_matrices[:,0]

    src_rmaps = torch.unbind(rmaps, dim=1)[1:]
    src_rots = torch.unbind(rot_matrices, dim=1)[1:]

    if (BS == 2) and (ref_rot.is_cuda == True):
        # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
        # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
        # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
        inv_ref_rot = torch.stack([torch.inverse(m) for m in ref_rot], dim=0)
    else:
        inv_ref_rot = torch.inverse(ref_rot)

    # reference view
    nx = normal_volume[:,0]
    ny = -normal_volume[:,1]
    nz = -torch.sqrt(torch.clamp(1 - nx**2 - ny**2, 0, 1))
    grid = torch.stack([nx,ny], dim=4).view(BS,D*H,W,2)
    ref_volume = F.grid_sample(ref_rmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    radiance_volumes = [ref_volume.view(BS,-1,D,H,W),]

    # source views
    for src_rmap, src_rot in zip(src_rmaps, src_rots):
        rot = src_rot @ inv_ref_rot # [BS,3,3]
        nv_rotated = torch.sum(rot[:,:,:,None,None,None] * torch.stack([nx,ny,nz], dim=1)[:,None,:,:,:,:], dim=2)
        nx_ = nv_rotated[:,0] * (nv_rotated[:,2] < 0.0).float()
        ny_ = nv_rotated[:,1] * (nv_rotated[:,2] < 0.0).float()
        m_ = (nv_rotated[:,2] < 0.0).float()

        grid = torch.stack([nx_,ny_], dim=4).view(BS,D*H,W,2)
        src_volume = F.grid_sample(src_rmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        radiance_volumes.append(src_volume.view(BS,-1,D,H,W) * m_[:,None,:,:,:])

    radiance_volumes = torch.stack(radiance_volumes, dim=1)
    return radiance_volumes

def construct_image_feature_volumes(feas, proj_matrices, depth_values):
    num_depth = depth_values.shape[1]
    ref_fea = feas[:,0]
    ref_proj = proj_matrices[:,0]

    src_feas = torch.unbind(feas, dim=1)[1:]
    src_projs = torch.unbind(proj_matrices, dim=1)[1:]

    fea_volumes = [ ref_fea[:,:,None,:,:].repeat(1,1,num_depth,1,1), ]
    for src_fea, src_proj in zip(src_feas, src_projs):
        src_fea_warped = homo_warping(src_fea, src_proj, ref_proj, depth_values)
        fea_volumes.append(src_fea_warped)

    fea_volumes = torch.stack(fea_volumes, dim=1)

    return fea_volumes # [BS,N,C,D,H,W]

# depth_prob_volume: [BS,D,H,W]
# depth_values: [BS,D]
# ret: [BS,H,W]
def sample_from_depth_prob_volume(depth_prob_volume, depth_values):
    BS,D,H,W = depth_prob_volume.size()
    dtype = depth_prob_volume.dtype
    device = depth_prob_volume.device

    depth_cdf = torch.stack(torch.unbind(torch.cumsum(depth_prob_volume, dim=1), dim=1), dim=-1) # [BS,H,W,D]
    z = torch.rand_like(depth_prob_volume[:,0], dtype = dtype, device=device) # [BS,H,W]
    sampled_indices = torch.searchsorted(depth_cdf, z[:,:,:,None], right=True)[..., 0] # [BS,H,W]
    sampled_indices = torch.clamp(sampled_indices, 0, D-1)

    d0 = torch.gather(depth_values[:,:,None,None].repeat(1,1,H,W), 1, sampled_indices[:,None,:,:])[:,0] # [BS,H,W]
    log_d0 = torch.log(d0)

    log_r = torch.log(torch.sqrt(depth_values[:,1] / depth_values[:,0])[:,None,None]) # [BS,1,1]
    log_d = log_d0 - log_r + 2 * log_r * torch.rand_like(log_d0, device=device)
    d = torch.exp(log_d)

    return d

def sample_from_normal_volume(normal_volume, depth_map, depth_values):
    D = depth_values.size(1)

    log_r = torch.log(torch.sqrt(depth_values[:,1] / depth_values[:,0])[:,None,None]) # [BS,1,1]
    indices = (torch.log(torch.clamp(depth_map / depth_values[:,0][:,None,None], 1e-12, None)) + log_r) / (2 * log_r) # range: [0,D)
    indices = torch.clamp(indices.long(), 0, D-1)[:,None,None,:,:].repeat(1,3,1,1,1) # [BS,3,1,H,W]
    normal = torch.gather(normal_volume, 2, indices)[:,:,0] # [BS,3,H,W]

    return normal

# depth_prob_volume: BS,D,H,W
# log_img_volumes: BS,N,3,D,H,W
# log_rmap_volumes: BS,N,3,D,H,W
# mask: BS,1,H,W
def compute_optimal_color(log_img_volumes, log_rmap_volumes, depth_prob_volume, mask):
    BS,N,C,D,H,W = log_img_volumes.size()
    depth_prob_volume = depth_prob_volume.detach() * mask

    w = depth_prob_volume[:,None].repeat(1,C*N,1,1,1).view(BS*C,-1) # [BS*C,N*D*H*W]
    x = torch.stack(torch.unbind(log_img_volumes - log_rmap_volumes, dim=2), dim=1).view(BS*C,-1) # [BS*C,N*D*H*W]

    x_sorted, sorted_indices = torch.sort(x, dim=1)
    w_sorted = torch.gather(w, 1, sorted_indices)

    cumsum_w = torch.cumsum(w_sorted, dim=1)
    l_prime =  2 * cumsum_w - cumsum_w[:,-1:]
    argmin_idx = torch.sum((l_prime < 0.0).long(), dim=1, keepdim=True)
    argmin_c = torch.gather(x_sorted, 1, argmin_idx) # [BS*C, 1]
    min_l = torch.sum(w_sorted * torch.abs(x_sorted - argmin_c), dim=1)

    optimal_log_color = argmin_c.view(BS,C)
    min_loss = min_l / torch.sum(mask) / N

    return optimal_log_color, min_loss

def compute_gradient_volume(depth_prob_volume, mask, intrinsic, depth_values, eps = 1e-9):
    dtype = depth_prob_volume.dtype
    device = depth_prob_volume.device
    BS,D,H,W = depth_prob_volume.size()

    # compute gradient of depth_prob_volume
    grad_u = 0.5 * (torch.roll(depth_prob_volume, -1, 3) - torch.roll(depth_prob_volume, 1, 3))
    grad_v = 0.5 * (torch.roll(depth_prob_volume, -1, 2) - torch.roll(depth_prob_volume, 1, 2))
    grad_d = (torch.roll(depth_prob_volume, -1, 1) - torch.roll(depth_prob_volume, 1, 1)) / (torch.roll(depth_values, -1, 1) - torch.roll(depth_values, 1, 1))[:,:,None,None]

    u = torch.arange(W, dtype=dtype, device=device)[None,None,None,:]
    v = torch.arange(H, dtype=dtype, device=device)[None,None,:,None]
    d = depth_values[:,:,None,None]

    grad_mx = 1.0 / d * grad_u
    grad_my = 1.0 / d * grad_v
    grad_mz = grad_d - u / d * grad_u - v / d * grad_v

    grad_m = torch.stack([grad_mx, grad_my, grad_mz], dim=1) # [BS,3,D,H,W]
    grad_xyz = torch.sum(intrinsic[:,:3,:3,None,None,None] * grad_m[:,:,None], dim=1)
    grad = grad_xyz * torch.tensor([1.0, -1.0, -1.0], dtype=dtype, device=device)[None,:,None,None,None]
    l = torch.sqrt(torch.sum(grad**2, dim=1, keepdim=True))
    return grad / (l + eps) * mask[:,None,:,:,:]

# log_img_volumes: BS,N,3,D,H,W
# log_rmaps: BS,N,3,Hn,Wn
# grad_volume: BS,3,D,H,W
# normal_volume: BS,3,D,H,W
# occlusion_masks: BS,N,1,H,W
@cuda.jit
def refine_normal_volume_kernel(
    log_img_volumes, 
    log_rmaps, 
    normal_volume, 
    occlusion_masks, 
    mask, 
    threshold_angle, 
    refined_normal_volume
):
    BS,N,C,D,H,W = log_img_volumes.shape
    Hn,Wn = log_rmaps.shape[3:5]

    cosine_threshold = math.cos(threshold_angle)

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    idx_depth = idx_thread
    while (mask[idx_batch,0,idx_v,idx_u] > 0.0) and (idx_depth < D):
        n_est = normal_volume[idx_batch,:,idx_depth,idx_v,idx_u]

        ax = math.acos(n_est[0])
        nx_min = math.cos(clip_device(ax + threshold_angle, 0, np.float32(np.pi)))
        nx_max = math.cos(clip_device(ax - threshold_angle, 0, np.float32(np.pi)))
        ay = math.acos(n_est[1])
        ny_min = math.cos(clip_device(ay + threshold_angle, 0, np.float32(np.pi)))
        ny_max = math.cos(clip_device(ay - threshold_angle, 0, np.float32(np.pi)))

        un_start = max(0, int(0.5 * (nx_min + 1) * Wn - 0.5))
        un_end = min(int(0.5 * (nx_max + 1) * Wn - 0.5) + 1, Wn-1)

        vn_start = max(0, int(0.5 * (-ny_max + 1) * Hn - 0.5))
        vn_end = min(int(0.5 * (-ny_min + 1) * Hn - 0.5) + 1, Hn-1)

        min_error = 1e12
        argmin_normal = (0.0, 0.0, 0.0)
        for idx_vn in range(vn_start, vn_end):
            for idx_un in range(un_start, un_end):
                nx = 2.0 * (idx_un + 0.5) / Wn - 1.0
                ny = -(2.0 * (idx_vn + 0.5) / Hn - 1.0)
                if (nx**2 + ny**2) >= 1.0:
                    continue
                nz = math.sqrt(1.0 - nx**2 - ny**2)

                cosine = nx * n_est[0] + ny * n_est[1] + nz * n_est[2]

                if cosine < cosine_threshold:
                    continue

                e = 0.0
                for idx_view in range(N):
                    if occlusion_masks[idx_batch,idx_view,0,idx_v,idx_u] > 0.5:
                        log_i = log_img_volumes[idx_batch,idx_view,:,idx_depth,idx_v,idx_u]
                        log_r = log_rmaps[idx_batch,idx_view,:,idx_vn,idx_un]
                        for idx_ch in range(C):
                            e += abs(log_i[idx_ch] - log_r[idx_ch]) / (N)

                if e < min_error:
                    min_error = e
                    argmin_normal = (nx,ny,nz)

        for idx_ch in range(3):
            refined_normal_volume[idx_batch,idx_ch,idx_depth,idx_v,idx_u] = argmin_normal[idx_ch]

        idx_depth += blockdim


# depth_prob_volume: BS,D,H,W
# log_img_volumes: BS,N,3,D,H,W
# log_rmaps: BS,N,3,Hn,Wn
# rot_matrices: BS,N,3,3
# intrinsic: BS,3,3
def refine_normal_volume(
    imgs, 
    rmaps, 
    normal_volume, 
    occlusion_masks,
    mask, 
    proj_matrices, 
    rot_matrices, 
    depth_values, 
    threshold_angle=0.5*0.17453288888 
):
    img_volumes = construct_image_feature_volumes(imgs, proj_matrices, depth_values) # [BS,N,3,D,H,W]
    log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None))
    log_rmaps = torch.log1p(torch.clamp(1000 * rmaps, 0, None))

    BS,N,C,D,H,W = log_img_volumes.size()
    device = log_img_volumes.device

    # rotate log_rmaps
    ref_rot = rot_matrices[:,0]
    rotated_log_rmaps = []
    for log_rmap, src_rot in zip(torch.unbind(log_rmaps, dim=1), torch.unbind(rot_matrices, dim=1)):
        rotated_log_rmaps.append(rotate_rmap(log_rmap, ref_rot, src_rot))
    log_rmaps = torch.stack(rotated_log_rmaps, dim=1) # BS,N,3,Hn,Wn

    # exhaustive search
    refined_normal_volume = torch.zeros((BS,3,D,H,W), device=device)
    num_threads = min(D, 256)
    refine_normal_volume_kernel[(BS,H,W), (num_threads)](
        log_img_volumes, 
        log_rmaps, 
        normal_volume, 
        occlusion_masks,
        mask, 
        np.float32(threshold_angle), 
        refined_normal_volume
    )

    return refined_normal_volume

# depth_prob_volume: BS,D,H,W
# log_img_volumes: BS,N,3,D,H,W
# log_rmaps: BS,N,3,Hn,Wn
# rot_matrices: BS,N,3,3
# intrinsic: BS,3,3
def refine_normal(
    imgs, 
    rmaps, 
    depth, 
    normal, 
    occlusion_masks,
    mask, 
    proj_matrices, 
    rot_matrices, 
    threshold_angle=0.17453288888 
):
    img_volumes = sample_from_src_views(depth, imgs, proj_matrices)[0][:,:,:,None,:,:]
    log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None))
    log_rmaps = torch.log1p(torch.clamp(1000 * rmaps, 0, None))

    normal_volume = normal[:,:,None,:,:]

    BS,N,C,D,H,W = log_img_volumes.size()
    device = log_img_volumes.device

    # rotate log_rmaps
    ref_rot = rot_matrices[:,0]
    rotated_log_rmaps = []
    for log_rmap, src_rot in zip(torch.unbind(log_rmaps, dim=1), torch.unbind(rot_matrices, dim=1)):
        rotated_log_rmaps.append(rotate_rmap(log_rmap, ref_rot, src_rot))
    log_rmaps = torch.stack(rotated_log_rmaps, dim=1) # BS,N,3,Hn,Wn

    # exhaustive search
    refined_normal_volume = torch.zeros((BS,3,D,H,W), device=device)
    num_threads = min(D, 256)
    refine_normal_volume_kernel[(BS,H,W), (num_threads)](
        log_img_volumes, 
        log_rmaps, 
        normal_volume, 
        occlusion_masks,
        mask, 
        threshold_angle, 
        refined_normal_volume
    )

    return refined_normal_volume[:,:,0]

# device functions
@cuda.jit(device=True)
def clip_device(x, min_, max_):
    return min(max(min_, x), max_)

@cuda.jit(device=True)
def scale_vector_device(scale, v):
    return scale * v[0], scale * v[1], scale * v[2]

@cuda.jit(device=True)
def add_vector_device(v1, v2):
    return v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]

@cuda.jit(device=True)
def smoooth_l1_device(x, y, beta):
    abs_e = abs(x - y)
    if abs_e < beta:
        return 0.5 * abs_e**2 / beta
    else:
        return abs_e - 0.5 * beta

# log_imgs: BS,N,3,H,W
# log_rmaps: BS,N,3,Hn,Wn
# mask: BS,1,H,W
# est_normal: BS,3,H,W
@cuda.jit
def find_argmin_depth_normal_kernel(
    log_imgs,
    log_rmaps,
    mask,
    proj_matrices,
    est_depth,
    est_normal,
    threshold_angle,
    M,
    beta, 
    depth_interval,
    num_depth,
    argmin_depth,
    argmin_normal
):
    BS,N,C,H,W = log_imgs.shape
    Hn,Wn = log_rmaps.shape[3:5]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    cosine_threshold = math.cos(threshold_angle)

    err_buf = cuda.local.array(shape=(10,), dtype=numba.float32)
    M = min(M, len(err_buf))
    
    if (mask[idx_batch,0,idx_v,idx_u] > np.float32(0.0)):
        d_est = est_depth[idx_batch,0,idx_v,idx_u]
        n_est = est_normal[idx_batch,:,idx_v,idx_u]

        ax = math.acos(n_est[0])
        nx_min = math.cos(clip_device(ax + threshold_angle, 0, np.float32(np.pi)))
        nx_max = math.cos(clip_device(ax - threshold_angle, 0, np.float32(np.pi)))
        ay = math.acos(n_est[1])
        ny_min = math.cos(clip_device(ay + threshold_angle, 0, np.float32(np.pi)))
        ny_max = math.cos(clip_device(ay - threshold_angle, 0, np.float32(np.pi)))

        un_start = max(0, int(0.5 * (nx_min + 1) * Wn - 0.5))
        un_end = min(int(0.5 * (nx_max + 1) * Wn - 0.5) + 1, Wn-1)

        vn_start = max(0, int(0.5 * (-ny_max + 1) * Hn - 0.5))
        vn_end = min(int(0.5 * (-ny_min + 1) * Hn - 0.5) + 1, Hn-1)

        min_sum_err = 1e12
        result = (0.0, 0.0, 0.0)
        result_depth = 0.0
        for idx_vn in range(vn_start, vn_end):
            for idx_un in range(un_start, un_end):
                nx = 2.0 * (idx_un + 0.5) / Wn - 1.0
                ny = -(2.0 * (idx_vn + 0.5) / Hn - 1.0)
                if (nx**2 + ny**2) >= 1.0:
                    continue
                nz = math.sqrt(1.0 - nx**2 - ny**2)

                cosine = nx * n_est[0] + ny * n_est[1] + nz * n_est[2]

                if cosine < cosine_threshold:
                    continue

                for idx_depth in range(num_depth):
                    d = d_est + np.float32(idx_depth - (num_depth // 2)) * depth_interval
                    m = ((idx_u + np.float32(0.5)) * d, (idx_v + np.float32(0.5)) * d, d)
                    for idx_view in range(N):
                        proj = proj_matrices[idx_batch, idx_view]
                        d_src = proj[2,0] * m[0] + proj[2,1] * m[1] + proj[2,2] * m[2] + proj[2,3]
                        u_src = (proj[0,0] * m[0] + proj[0,1] * m[1] + proj[0,2] * m[2] + proj[0,3]) / d_src - 0.5
                        v_src = (proj[1,0] * m[0] + proj[1,1] * m[1] + proj[1,2] * m[2] + proj[1,3]) / d_src - 0.5

                        if (u_src < 0.0) or (v_src < 0.0) or (u_src >= (W-1)) or (v_src >= (H-1)):
                            log_i = (0.0, 0.0, 0.0)
                        else:
                            u_src_mod = u_src - int(u_src)
                            v_src_mod = v_src - int(v_src)
                            log_i_a = log_imgs[idx_batch,idx_view,:,int(v_src),int(u_src)]
                            log_i_b = log_imgs[idx_batch,idx_view,:,int(v_src),int(u_src)+1]
                            log_i_c = log_imgs[idx_batch,idx_view,:,int(v_src)+1,int(u_src)]
                            log_i_d = log_imgs[idx_batch,idx_view,:,int(v_src)+1,int(u_src)+1]

                            log_i_ab = add_vector_device(scale_vector_device(1.0-u_src_mod, log_i_a), scale_vector_device(u_src_mod, log_i_b))
                            log_i_cd = add_vector_device(scale_vector_device(1.0-u_src_mod, log_i_c), scale_vector_device(u_src_mod, log_i_d))
                            log_i = add_vector_device(scale_vector_device(1.0-v_src_mod, log_i_ab), scale_vector_device(v_src_mod, log_i_cd))

                        log_r = log_rmaps[idx_batch,idx_view,:,idx_vn,idx_un]

                        e = 0.0
                        for idx_ch in range(C):
                            e += smoooth_l1_device(log_i[idx_ch], log_r[idx_ch], beta)

                        idx_back = min(idx_view, M - 1)

                        if (idx_view < M) or (e < err_buf[idx_back]):
                            err_buf[idx_back] = e

                            # bubble sort
                            for idx_front in range(1,idx_back):
                                for j in range(idx_back-1,idx_front-1,-1):
                                    if err_buf[j] > err_buf[j+1]:
                                        tmp = err_buf[j]
                                        err_buf[j] = err_buf[j+1]
                                        err_buf[j+1] = tmp
                    sum_err = 0.0
                    for i in range(M):
                        sum_err += err_buf[i]
                
                    if sum_err < min_sum_err:
                        min_sum_err = sum_err
                        result = (nx,ny,nz)
                        result_depth = d

        for idx_ch in range(3):
            argmin_normal[idx_batch,idx_ch,idx_v,idx_u] = result[idx_ch]
        argmin_depth[idx_batch,0,idx_v,idx_u] = result_depth

def find_argmin_depth_normal(
    log_imgs,
    log_rmaps,
    proj_matrices,
    rot_matrices, 
    mask,
    est_depth,
    est_normal, 
    threshold_angle=0.262, 
    M=3, 
    beta=0.3,
    depth_interval = 0.001,
    num_depth = 9,
):
    BS,_,H,W = est_normal.size()
    # rotate log_rmaps
    ref_rot = rot_matrices[:,0]
    rotated_log_rmaps = []
    for log_rmap, src_rot in zip(torch.unbind(log_rmaps, dim=1), torch.unbind(rot_matrices, dim=1)):
        rotated_log_rmaps.append(rotate_rmap(log_rmap, ref_rot, src_rot))
    log_rmaps = torch.stack(rotated_log_rmaps, dim=1) # BS,N,3,Hn,Wn

    proj_matrices = proj_matrices @ torch.inverse(proj_matrices[:,0:1])

    argmin_depth = torch.zeros_like(est_depth, dtype=est_normal.dtype, device=est_normal.device)
    argmin_normal = torch.zeros_like(est_normal, dtype=est_normal.dtype, device=est_normal.device)

    find_argmin_depth_normal_kernel[(BS,H,W),(1,)](
        log_imgs.detach(), 
        log_rmaps.detach(), 
        mask.detach(), 
        proj_matrices.detach(),
        est_depth.detach(),
        est_normal.detach(), 
        threshold_angle, 
        M,
        beta,
        depth_interval,
        num_depth,
        argmin_depth,
        argmin_normal
    )

    return argmin_depth, argmin_normal

def search_optimal_scale(mvsfsnet, imgs, rmaps, masks, proj_matrices, rot_matrices, intrinsics, depth_values):
    assert imgs.size(0) == 1
    dn_consistency_loss = DepthNormalConsistencyLoss()
    list_argmin_scale = []
    for idx_ch in range(3):
        imgs_gray = imgs[:,:,idx_ch:idx_ch+1].repeat(1,1,3,1,1)
        rmaps_gray = rmaps[:,:,idx_ch:idx_ch+1].repeat(1,1,3,1,1)
        list_scale = 1.1**np.arange(-20.0, 21.0, 1.0)
        list_loss = []
        list_loss_recon = []
        list_loss_dn = []
        for scale in list_scale:
            out = mvsfsnet(imgs_gray, scale * rmaps_gray, rot_matrices, proj_matrices, depth_values)
            depth_prob_volume = out['depth_prob_volume'] * masks[:,0,0,None]
            normal_volume = out['normal_volume'] * masks[:,0,:,None]

            # compute image reconstruction loss
            img_volumes = construct_image_feature_volumes(imgs_gray, proj_matrices, depth_values) # [BS,N,3,D,H,W]
            log_img_volumes = torch.log1p(torch.clamp(1000 * img_volumes, 0, None))
            rmap_volumes = construct_reflectance_feature_volumes(scale * rmaps_gray, rot_matrices, normal_volume) # [BS,N,3,D,H,W]
            log_rmap_volumes = torch.log1p(torch.clamp(1000 * rmap_volumes, 0, None))
            log_error_volumes = torch.mean(torch.abs(log_rmap_volumes - log_img_volumes), dim=2) # [BS,N,D,H,W]
            log_error_volume = torch.mean(log_error_volumes, dim=1) * masks[:,0,:] # [BS,D,H,W]
            loss_recon = torch.sum(torch.sum(log_error_volume * depth_prob_volume, dim=1) * masks[:,0,0]) / torch.sum(masks[:,0,0])

            # compute depth-normal consistency loss
            loss_dn = 0.01 * dn_consistency_loss(depth_prob_volume, normal_volume, masks[:,0], intrinsics[:,0], depth_values)

            loss = loss_recon + loss_dn
            print(scale, loss.item())

            list_loss.append(loss.item())
            list_loss_recon.append(loss_recon.item())
            list_loss_dn.append(loss_dn.item())
        argmin_scale = list_scale[np.argmin(list_loss)]
        list_argmin_scale.append(argmin_scale)
        if False:
            import matplotlib.pyplot as plt
            plt.close()
            plt.semilogx(list_scale, list_loss)
            plt.semilogx(list_scale, list_loss_recon)
            plt.semilogx(list_scale, list_loss_dn)
            plt.semilogx([argmin_scale,argmin_scale], [0.0, np.max(list_loss)], 'blue', linestyle='dashed')
            plt.semilogx([1.0, 1.0], [0.0, np.max(list_loss)], 'red', linestyle='dashed')
            plt.grid()
            plt.show()
    argmin_scale = torch.tensor(list_argmin_scale, dtype=imgs.dtype, device=imgs.device)[None,:]
    print('optimal_scale:', argmin_scale)
    return argmin_scale

def depth_from_silhouette(ref_mask, ref_proj, ref_intrinsic, ref_depth_range, src_masks, src_projs, numdepth=192):
    depth_values = get_depth_values(
        torch.mean(ref_depth_range, dim=1), 
        ref_mask.size()[2:4], 
        ref_intrinsic, 
        numdepth=numdepth
    )

    masks = torch.cat([ref_mask[:,None], src_masks], dim=1)
    proj_matrices = torch.cat([ref_proj[:,None], src_projs], dim=1)

    volume_mask = (torch.prod(construct_image_feature_volumes(masks, proj_matrices, depth_values), dim=1)[:,0] > 0.0).float() # [BS,D,H,W]
    depth_volume = volume_mask * depth_values[:,:,None,None]
    depth_volume[volume_mask == 0.0] = 1e12
    depth = torch.min(depth_volume, dim=1)[0][:,None]
    mask = (torch.sum(volume_mask, dim=1, keepdim=True) > 0.0).float()
    depth = depth * mask

    return depth, mask
