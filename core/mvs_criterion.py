import torch
import torch.nn.functional as F

class DepthLogLikelihoodLoss(torch.nn.Module):
    def __init__(self):
        super(DepthLogLikelihoodLoss, self).__init__()

    # depth_prob_volume: BS,D,H,W
    # gt_depth: BS,1,H,W
    # mask: BS,1,H,W
    # depth_values: BS,D
    def forward(self, depth_prob_volume, gt_depth, mask, depth_values):
        D = depth_values.size(1)

        log_r = torch.log(torch.sqrt(depth_values[:,1] / depth_values[:,0])[:,None,None]) # [BS,1,1]
        indices = (torch.log(torch.clamp(gt_depth[:,0] / depth_values[:,0][:,None,None], 1e-12, None)) + log_r) / (2 * log_r) # [BS,H,W], range: [0,D)
        indices = torch.clamp(indices.long(), 0, D-1)[:,None,:,:] # [BS,1,H,W]
        p = torch.gather(depth_prob_volume, 1, indices) # [BS,1,H,W]
        return torch.sum(-torch.log(p + 1e-6) * mask) / torch.sum(mask)

class DepthNormalConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(DepthNormalConsistencyLoss, self).__init__()

    # for debugging
    def get_gradient_volume(self, depth_prob_volume, mask, intrinsic, depth_values, eps = 1e-9):
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


    # depth_prob_volume: BS,D,H,W
    # normal_volume: BS,3,D,H,W
    # mask: BS,1,H,W
    # intrinsic: BS,3,3
    # depth_values: BS,D
    def forward(self, depth_prob_volume, normal_volume, mask, intrinsic, depth_values, eps = 1e-9):
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

        dp2 = torch.sum(grad * normal_volume, dim=1)**2
        l2 = torch.sum(grad**2, dim=1)
        ndp = torch.sqrt((dp2 + eps) / (l2 + eps))
        error_volume = torch.arccos(torch.clamp(ndp, 0, 0.9999)) # [BS,D,H,W]

        mask_volume = torch.ones_like(depth_prob_volume, dtype=dtype, device=device)
        #mask_volume[:,0] = 0.0
        #mask_volume[:,-1] = 0.0
        mask_volume[:,:,0] = 0.0
        mask_volume[:,:,-1] = 0.0
        mask_volume[:,:,:,0] = 0.0
        mask_volume[:,:,:,-1] = 0.0

        error_map = torch.sum(error_volume * mask_volume * depth_prob_volume, dim=1, keepdim=True) # [BS,1,H,W]
        return torch.sum(error_map * mask) / torch.sum(mask)
