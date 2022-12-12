import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ConvBnReLU3D, depth_regression, homo_warping
from .sfs_models import SfSNet, UNet, pad_invalid_values
from .mvs_utils import rotate_prob_volume

class CostVolumeFilteringNet(nn.Module):
    def __init__(self, num_fea_in=32, num_fea_out=1):
        super(CostVolumeFilteringNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.downsample = nn.AvgPool3d(2, stride=2)
        
        self.conv0 = ConvBnReLU3D(num_fea_in, 8)

        self.conv1 = ConvBnReLU3D(8, 16)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = ConvBnReLU3D(64, 32)

        self.conv9 = ConvBnReLU3D(32, 16)

        self.conv11 = ConvBnReLU3D(16, 8)

        self.prob = nn.Conv3d(8, num_fea_out, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(self.downsample(conv0)))
        conv4 = self.conv4(self.conv3(self.downsample(conv2)))
        x = self.conv6(self.conv5(self.downsample(conv4)))
        x = conv4 + self.upsample(self.conv7(x))
        x = conv2 + self.upsample(self.conv9(x))
        x = conv0 + self.upsample(self.conv11(x))
        x = self.prob(x)
        return x

class MVSfSNet(nn.Module):
    def __init__(self, wo_sfs=False):
        super(MVSfSNet, self).__init__()
        self.wo_sfs = wo_sfs
        if wo_sfs:
            self.feanet_i = UNet(3,32)
        else:
            self.sfsnet = SfSNet()
            self.feanet = nn.Sequential(
                nn.Conv3d(5,64,1),
                nn.LeakyReLU(),
                nn.Conv3d(64,64,1),
                nn.LeakyReLU(),
                nn.Conv3d(64,64,1),
                nn.LeakyReLU(),
                nn.Conv3d(64,16,1),
            )
            self.feanet_i = UNet(22,32)
        self.costnet = nn.Sequential(
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,32,1),
        )
        self.cv_filtering = CostVolumeFilteringNet(num_fea_out=3)

    # log_hdr_img:  (BS,N,3,H,W), log-rgb(-k~+k)+mask(0or1)
    # log_hdr_rmap: (BS,N,3,H,W), log-rgb(-k~+k)+mask(0or1)
    # rot_matrices: (BS,N,3,3)
    # proj_matrices: (BS,N,4,4)
    # depth_values: (BS,D)
    def forward(self, hdr_imgs, hdr_rmaps, rot_matrices, proj_matrices, depth_values):
        # step 1. normal estimation & feature extraction
        hdr_imgs = torch.unbind(hdr_imgs, dim=1)
        hdr_rmaps = torch.unbind(hdr_rmaps, dim=1)
        rot_matrices = torch.unbind(rot_matrices, dim=1)
        proj_matrices = torch.unbind(proj_matrices, dim=1)

        assert len(hdr_imgs) == len(hdr_rmaps) == len(proj_matrices) == len(rot_matrices), "Different number of images and projection matrices"
        img_height, img_width = hdr_imgs[0].shape[2], hdr_imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(hdr_imgs)
        device = hdr_imgs[0].device

        ref_rot = rot_matrices[0]
        features = []
        for hdr_img, hdr_rmap, src_rot in zip(hdr_imgs, hdr_rmaps, rot_matrices):
            if self.wo_sfs:
                # image feature
                hdr_img = pad_invalid_values(hdr_img)
                mask = (torch.max(hdr_img, dim=1, keepdim=True)[0] > 0.0).float()
                mean_color = torch.sum(hdr_img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
                log_img = torch.log1p(100.0 * hdr_img / mean_color[:,:,None,None]) * mask
                pixel_fea = self.feanet_i(log_img) # [BS,16,H,W]
            else:
                out = self.sfsnet(hdr_img, hdr_rmap)[-1]
                hypotheses_normal = out['hypotheses_normal'] # [BS,3,N,H,W]
                hypotheses_likelihood = out['hypotheses_likelihood'] # [BS,1,N,H,W]
                hypotheses_prior = out['hypotheses_prior'] # [BS,1,N,H,W]

                hypotheses_likelihood = hypotheses_likelihood / torch.clamp(torch.max(hypotheses_likelihood, dim=2, keepdim=True)[0], 1e-9, None)
                hypotheses_prior = hypotheses_prior / torch.clamp(torch.max(hypotheses_prior, dim=2, keepdim=True)[0], 1e-9, None)

                # rotation
                if (src_rot.size(0) == 2) and (ref_rot.is_cuda == True):
                    # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
                    # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
                    # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
                    inv_src_rot = torch.stack([torch.inverse(m) for m in src_rot], dim=0)
                    rot = torch.matmul(ref_rot, inv_src_rot)
                else:
                    rot = torch.matmul(ref_rot, torch.inverse(src_rot))
                n_ = hypotheses_normal * torch.tensor([1.0, -1.0, -1.0], device=device)[None,:,None,None,None]
                n_ = torch.sum(rot[:,:,:,None,None,None] * n_[:,None,:], dim=2)
                hypotheses_normal = n_ * torch.tensor([1.0, -1.0, -1.0], device=device)[None,:,None,None,None]

                # per-hypothesis feature extraction
                hypotheses = torch.cat([hypotheses_normal, hypotheses_likelihood, hypotheses_prior], dim=1) # [BS,5,N,H,W]
                hypotheses_fea = self.feanet(hypotheses)

                # bypass for effectively leveraging normals with the highest probability
                argmax_normals = hypotheses[:,:3,0,:,:] # [BS,3,H,W]

                # inter-hypothesis feature pooling
                pixel_pdf_fea = torch.max(hypotheses_fea, dim=2)[0] # [BS,16,H,W]

                # image feature
                hdr_img = pad_invalid_values(hdr_img)
                mask = (torch.max(hdr_img, dim=1, keepdim=True)[0] > 0.0).float()
                mean_color = torch.sum(hdr_img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
                log_img = torch.log1p(100.0 * hdr_img / mean_color[:,:,None,None]) * mask
                pixel_fea = self.feanet_i(torch.cat([log_img, argmax_normals, pixel_pdf_fea], dim=1)) # [BS,16,H,W]

            features.append(pixel_fea)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]     

        # latent cost volume construction
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        cost_volume = 0
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            fea_volume = torch.cat([ref_volume, src_volume], dim=1)
            cost_volume = cost_volume + self.costnet(fea_volume) / len(src_features)
            del src_volume
            del fea_volume
        del ref_volume

        # cost volume filtering
        filtered_volume = self.cv_filtering(cost_volume)

        # compute depth probability
        depth_prob_volume = F.softmax(filtered_volume[:,0:1].squeeze(1), dim=1)
        depth = depth_regression(depth_prob_volume, depth_values=depth_values)

        # compute normal
        p,q = torch.unbind(filtered_volume[:,1:3], dim=1)
        l = torch.sqrt(p**2 + q**2 + 1.0)
        normal_volume = torch.stack([p/l,q/l,1.0/l], dim=1) # [BS,3,D,H,W]
        normal = torch.sum(normal_volume * depth_prob_volume[:,None], dim=2) # [BS,3,H,W]
        l = torch.sqrt(torch.clamp(torch.sum(normal**2, dim=1, keepdim=True), 1e-2, None))
        normal = normal / l

        return {
            'depth': depth,
            'normal': normal,
            'depth_prob_volume': depth_prob_volume,
            'normal_volume': normal_volume,
        }

    # log_hdr_img:  (BS,N,3,H,W), log-rgb(-k~+k)+mask(0or1)
    # log_hdr_rmap: (BS,N,3,H,W), log-rgb(-k~+k)+mask(0or1)
    # rot_matrices: (BS,N,3,3)
    # proj_matrices: (BS,N,4,4)
    # depth_values: (BS,D)
    def get_intermediate_features(self, hdr_imgs, hdr_rmaps, rot_matrices, proj_matrices, depth_values):
        # step 1. normal estimation & feature extraction
        hdr_imgs = torch.unbind(hdr_imgs, dim=1)
        hdr_rmaps = torch.unbind(hdr_rmaps, dim=1)
        rot_matrices = torch.unbind(rot_matrices, dim=1)
        proj_matrices = torch.unbind(proj_matrices, dim=1)

        assert len(hdr_imgs) == len(hdr_rmaps) == len(proj_matrices) == len(rot_matrices), "Different number of images and projection matrices"
        img_height, img_width = hdr_imgs[0].shape[2], hdr_imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(hdr_imgs)
        device = hdr_imgs[0].device

        ref_rot = rot_matrices[0]
        features = []
        for idx_view, (hdr_img, hdr_rmap, src_rot) in enumerate(zip(hdr_imgs, hdr_rmaps, rot_matrices)):
            out = self.sfsnet(hdr_img, hdr_rmap)[-1]
            hypotheses_normal = out['hypotheses_normal'] # [BS,3,N,H,W]
            hypotheses_likelihood = out['hypotheses_likelihood'] # [BS,1,N,H,W]
            hypotheses_prior = out['hypotheses_prior'] # [BS,1,N,H,W]

            hypotheses_likelihood = hypotheses_likelihood / torch.clamp(torch.max(hypotheses_likelihood, dim=2, keepdim=True)[0], 1e-9, None)
            hypotheses_prior = hypotheses_prior / torch.clamp(torch.max(hypotheses_prior, dim=2, keepdim=True)[0], 1e-9, None)
            
            if idx_view == 1:
                sfs_normals = hypotheses_normal
                sfs_probs = hypotheses_prior * hypotheses_likelihood

            # rotation
            if (src_rot.size(0) == 2) and (ref_rot.is_cuda == True):
                # Avoid a bug of torch.inverse() in pytorch1.7.0+cuda11.0
                # https://github.com/pytorch/pytorch/issues/47272#issuecomment-722278640
                # This has alredy been fixed so that you can remove this when using Pytorch1.7.1>=
                inv_src_rot = torch.stack([torch.inverse(m) for m in src_rot], dim=0)
                rot = torch.matmul(ref_rot, inv_src_rot)
            else:
                rot = torch.matmul(ref_rot, torch.inverse(src_rot))
            n_ = hypotheses_normal * torch.tensor([1.0, -1.0, -1.0], device=device)[None,:,None,None,None]
            n_ = torch.sum(rot[:,:,:,None,None,None] * n_[:,None,:], dim=2)
            hypotheses_normal = n_ * torch.tensor([1.0, -1.0, -1.0], device=device)[None,:,None,None,None]

            if idx_view == 1:
                sfs_normals_rotated = hypotheses_normal

            # per-hypothesis feature extraction
            hypotheses = torch.cat([hypotheses_normal, hypotheses_likelihood, hypotheses_prior], dim=1) # [BS,5,N,H,W]
            hypotheses_fea = self.feanet(hypotheses)

            # inter-hypothesis feature pooling
            pixel_fea = torch.max(hypotheses_fea, dim=2)[0] # [BS,C,H,W]

            if idx_view == 1:
                pixel_fea_sfs = pixel_fea

            # image feature
            hdr_img = pad_invalid_values(hdr_img)
            mask = (torch.max(hdr_img, dim=1, keepdim=True)[0] > 0.0).float()
            mean_color = torch.sum(hdr_img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
            log_img = torch.log1p(100.0 * hdr_img / mean_color[:,:,None,None]) * mask
            pixel_fea_i = self.feanet_i(log_img) # [BS,C,H,W]

            if idx_view == 1:
                pixel_fea_img = pixel_fea_i

            pixel_fea = torch.cat([pixel_fea, pixel_fea_i], dim=1)

            features.append(pixel_fea)

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]     

        # latent cost volume construction
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        ref_fea_volume = ref_volume
        per_src_view_cost_volumes = []
        cost_volume = 0
        for idx_view, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            # warpped features
            src_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if idx_view == 1:
                src_fea_volume = src_volume
            fea_volume = torch.cat([ref_volume, src_volume], dim=1)
            per_src_view_cost_volumes.append(self.costnet(fea_volume))
            cost_volume = cost_volume + self.costnet(fea_volume) / len(src_features)
            #del src_volume
            #del fea_volume
        #del ref_volume

        # cost volume filtering
        filtered_volume = self.cv_filtering(cost_volume)

        # compute depth probability
        depth_prob_volume = F.softmax(filtered_volume[:,0:1].squeeze(1), dim=1)
        depth = depth_regression(depth_prob_volume, depth_values=depth_values)

        # compute normal
        p,q = torch.unbind(filtered_volume[:,1:3], dim=1)
        l = torch.sqrt(p**2 + q**2 + 1.0)
        normal_volume = torch.stack([p/l,q/l,1.0/l], dim=1) # [BS,3,D,H,W]
        normal = torch.sum(normal_volume * depth_prob_volume[:,None], dim=2) # [BS,3,H,W]
        l = torch.sqrt(torch.clamp(torch.sum(normal**2, dim=1, keepdim=True), 1e-2, None))
        normal = normal / l

        return {
            'sfs_normals': sfs_normals,
            'sfs_normals_rotated': sfs_normals_rotated,
            'sfs_probs': sfs_probs,
            'pixel_fea_sfs': pixel_fea_sfs,
            'pixel_fea_img': pixel_fea_img,
            'ref_fea': ref_feature,
            'src_fea': src_features[0],
            'ref_fea_volume': ref_fea_volume,
            'src_fea_volume': src_fea_volume,
            'per_src_view_cost_volumes': per_src_view_cost_volumes,
            'cost_volume': cost_volume,
            'depth': depth,
            'normal': normal,
            'depth_prob_volume': depth_prob_volume,
            'normal_volume': normal_volume,
        }