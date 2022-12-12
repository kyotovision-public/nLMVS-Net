import torch
from .sfs_utils import sample_normal_prob

class NormalLogLikelihoodLoss(torch.nn.Module):
    def __init__(self):
        super(NormalLogLikelihoodLoss, self).__init__()

    # hypotheses_probability: BS*1*N*H*W
    # grid_normal: BS*2*N*H*W
    # gt_normal: BS*3*H*W
    # mask: BS*1*H*W
    def forward(self, hypotheses_probability, grid_normal, patch_size, gt_normal, mask):
        e = (gt_normal[:,:2,None] - grid_normal) / patch_size
        e = e * torch.tensor([1.0, -1.0], device=e.device)[None,:,None,None,None]
        mask_gt = torch.prod((e >= 0.0) * (e < 1.0), dim=1, keepdim=True).float() # [BS,1,N,H,W]

        p = torch.max(mask_gt * hypotheses_probability, dim=2)[0] # [BS,1,H,W]

        if False:
            import matplotlib.pyplot as plt
            plt.close()
            g = plt.subplot(1,1,1)
            g.scatter(grid_normal[0,0,0:1,64,64].cpu(), grid_normal[0,1,0:1,64,64].cpu())
            g.scatter(gt_normal[0,0,64,64].cpu(), gt_normal[0,1,64,64].cpu())
            g.scatter(grid_normal[0,0,:,64,64][mask_gt[0,0,:,64,64].bool()].cpu(), grid_normal[0,1,:,64,64][mask_gt[0,0,:,64,64].bool()].cpu(), c='r')
            g.set_xlim([-1,1])
            g.set_ylim([-1,1])
            g.set_aspect('equal')  
            plt.show()

        #mask = (torch.sum(mask_gt, dim=2) > 0.0).float() * mask
        loss = torch.sum(-torch.log(p+1e-4) * mask) / (torch.sum(mask)+1e-9)
        return loss

class NormalAccuracy(torch.nn.Module):
    def __init__(self):
        super(NormalAccuracy, self).__init__()

    # hypotheses_probability: BS*1*N*H*W
    # grid_normal: BS*2*N*H*W
    # gt_normal: BS*3*H*W
    # mask: BS*1*H*W
    def forward(self, grid_normal, patch_size, gt_normal, mask):
        e = (gt_normal[:,:2] - grid_normal[:,:2,0]) / patch_size
        e = e * torch.tensor([1.0, -1.0], device=e.device)[None,:,None,None]
        mask_gt = torch.prod((e >= 0.0) * (e < 1.0), dim=1, keepdim=True).float() # [BS,1,H,W]
        return torch.sum(mask_gt * mask) / torch.sum(mask)