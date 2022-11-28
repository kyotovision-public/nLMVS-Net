import sys
import os
import subprocess

import numpy as np
import torch
from torch.utils.data.dataset import Subset

from glob import glob

import trimesh

from core.dataset import MVSRDataset
from est_shape_mat import est_shape_mat

import cv2
from eval_utils.visualize_merl import visualize_merl_as_sheres
import json

from core.ibrdf.ibrdf.util import LoadMERL

import argparse

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numdepth = 192
num_views_for_rfs = 3

parser = argparse.ArgumentParser()
parser.add_argument('object_name', help='name of the target object (e.g., 00152)', type=str)
parser.add_argument('view_index', help='index of the reference view', type=int)
parser.add_argument('--wo-sfs', action='store_true')
parser.add_argument('--wo-coarse', action='store_true')
parser.add_argument('--wo-fine', action='store_true')
parser.add_argument('--dataset-path', default=os.environ['HOME']+'/data/mvs_eval/rendered', type=str)
parser.add_argument('--exp-name', default='nlmvss', type=str)
#parser.add_argument('--wo-occlusion-detection', action='store_true')
args = parser.parse_args()

object_name = args.object_name
object_names = [s.split('/')[-1] for s in sorted(glob(args.dataset_path+'/?????'))]
object_id = object_names.index(object_name)

#object_id = args.object_id
view_index = args.view_index

dataset_path = args.dataset_path
num_images_per_object = len(glob(args.dataset_path+'/00000/cams/*_cam.txt'))

out_dir = './run/est_shape_mat_per_view'
if args.wo_sfs:
    out_dir = out_dir+'_wo_sfs'
else:
    if args.wo_coarse:
        out_dir = out_dir+'_wo_coarse'
    if args.wo_fine:
        out_dir = out_dir+'_wo_fine'
#elif args.wo_occlusion_detection:
#    out_dir = out_dir+'_wo_occlusion_detection'

out_dir = out_dir+'/'+args.exp_name+'/'+object_name+'/'+str(view_index).zfill(3)
os.makedirs(out_dir, exist_ok=True)

print('dataset path:', dataset_path)
print('object id:', object_id)
print('num_images_per_object:', num_images_per_object)
print('out_dir:', out_dir)

# dataset settings
dataset_options = {
    'num_neighbors': 4,
    'use_crop': True,
    'img_size': (128,128),
    'rmap_mode': 'sphere',
    'rmap_size': (128,128),
    'mask_img': True,
    'use_illum': True,
}
dataset = MVSRDataset(dataset_path, dataset_options)

list_split = np.arange(len(dataset))
test_subset_indices =  list_split[object_id*num_images_per_object:(object_id+1)*num_images_per_object]
test_dataset = Subset(dataset, test_subset_indices)

test_subset_indices =  [view_index,]
test_dataset = Subset(test_dataset, test_subset_indices)

# compute bbox diagonal of gt geometry
gt_mesh_files = sorted(glob(os.environ['HOME']+'/data/mvs_eval/assets/shape/*.obj'))
gt_mesh_file = gt_mesh_files[(object_id // 6) % len(gt_mesh_files)]

mesh = trimesh.load(gt_mesh_file)
verts = np.asarray(mesh.vertices)

bbox_min = np.min(verts, axis=0)
bbox_max = np.max(verts, axis=0)
#bbox_center = 0.5 * (bbox_min + bbox_max)
bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
camera_min_distance = 1.0
camera_tangent = 1.0 / (2.0 / 1200 * 5000.0)
scale = 0.5 * bbox_diagonal / camera_tangent / camera_min_distance

#verts -= bbox_center
#verts /= scale
bbox_diagonal /= scale
print('diagonal of gt mesh:', bbox_diagonal)

# compute coarse depth range
# currently computed from gt depth map
gt_depth = test_dataset[0]['gt_depths'][0,0]
d_min = torch.min(gt_depth[gt_depth > 0.0])
d_max = torch.max(gt_depth[gt_depth > 0.0])
est_depth_range = torch.stack([d_min, d_max])

est_shape_mat(
    test_dataset, 
    out_dir,
    device,
    num_views_for_rfs = num_views_for_rfs,
    numdepth=numdepth,
    num_iteration_per_epoch=25,
    bbox_diagonal=bbox_diagonal,
    wo_sfs=args.wo_sfs,
    occlusion_handling=False,#(not args.wo_occlusion_detection),
    wo_coarse=args.wo_coarse,
    wo_fine=args.wo_fine,
    est_depth_range_list=[est_depth_range],
)

# size of inputs: BS,3,90,90,180
def compute_error(est_brdf, gt_brdf):
    mask = torch.any(gt_brdf > 0, dim=1, keepdim=True).float()
    log_est_brdf = torch.log(torch.clamp(est_brdf,1e-4,None))
    log_gt_brdf = torch.log(torch.clamp(gt_brdf,1e-4,None))
    err = torch.sum((log_est_brdf - log_gt_brdf)**2 * mask) / (3 * torch.sum(mask))
    return err

# 
# visualize gt brdf as cascade spheres if exists
out_dir = out_dir+'/final_results'
if not args.wo_sfs:
    gt_brdf_files = sorted(glob(os.environ['HOME']+'/data/mvs_eval/assets/material/*.binary'))
    gt_brdf_file = gt_brdf_files[(object_id % 6)]

    img, mask = visualize_merl_as_sheres(gt_brdf_file)
    cv2.imwrite(out_dir+'/gt_brdf.exr', img[:,:,::-1])

    img = np.clip(img**(1/2.2), 0.0, 1.0)
    img = img / np.clip(np.max(img), 1e-9,1)
    img = np.concatenate([img,mask[:,:,None]], axis=-1)

    img[:,:,:3] = img[:,:,:3][:,:,::-1]
    cv2.imwrite(out_dir+'/gt_brdf.png', (255*img).astype(np.uint8))

    # 4 spheres
    img, mask = visualize_merl_as_sheres(gt_brdf_file, 45)
    cv2.imwrite(out_dir+'/gt_brdf_45.exr', img[:,:,::-1])

    img = np.clip(img**(1/2.2), 0.0, 1.0)
    img = img / np.clip(np.max(img), 1e-9,1)
    img = np.concatenate([img,mask[:,:,None]], axis=-1)

    img[:,:,:3] = img[:,:,:3][:,:,::-1]
    cv2.imwrite(out_dir+'/gt_brdf_45.png', (255*img).astype(np.uint8))

    subprocess.run([
        'python', 'eval_utils/create_log_error_map.py',
        out_dir+'/brdf_45.exr',
        out_dir+'/gt_brdf_45.exr',
        out_dir+'/brdf_45_error.png'
    ])

    # quantitative evaluation
    brdf_file = out_dir+'/brdf.binary'
    gt_brdf = LoadMERL(gt_brdf_file).float()[None].to(device)
    brdf = LoadMERL(brdf_file).float()[None].to(device)
    error = compute_error(brdf, gt_brdf)

    with open(out_dir+'/brdf_error.json', 'w') as f:
        json.dump(
            {'brdf_log_rmse': error.item()}, 
            f, 
            ensure_ascii=True
        )