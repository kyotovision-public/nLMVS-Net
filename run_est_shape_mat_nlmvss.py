import sys
import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset

from glob import glob

import trimesh

from core.dataset import MVSRDataset
from est_shape_mat import est_shape_mat

import argparse

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numdepth = 192
num_views_for_rfs = 3

parser = argparse.ArgumentParser()
parser.add_argument('object_name', help='name of the target object (e.g., 00152)', type=str)
parser.add_argument('--wo-sfs', action='store_true')
parser.add_argument('--wo-coarse', action='store_true')
parser.add_argument('--wo-fine', action='store_true')
parser.add_argument('--wo-occlusion-detection', action='store_true')
parser.add_argument('--dataset-path', default=os.environ['HOME']+'/data/mvs_eval/rendered', type=str)
parser.add_argument('--exp-name', default='nlmvss', type=str)
args = parser.parse_args()

object_name = args.object_name
object_names = [s.split('/')[-1] for s in sorted(glob(args.dataset_path+'/?????'))]
object_id = object_names.index(object_name)

#object_id = args.object_id

dataset_path = args.dataset_path
num_images_per_object = len(glob(args.dataset_path+'/00000/cams/*_cam.txt'))

out_dir = './run/est_shape_mat'
if args.wo_sfs:
    out_dir = out_dir+'_wo_sfs'
elif args.wo_occlusion_detection:
    out_dir = out_dir+'_wo_occlusion_detection'
elif args.wo_coarse:
    out_dir = out_dir+'_wo_coarse'
elif args.wo_fine:
    out_dir = out_dir+'_wo_fine'
out_dir = out_dir+'/'+args.exp_name+'/'+object_name
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

# compute bbox diagonal of gt geometry
gt_mesh_files = sorted(glob(os.environ['HOME']+'/data/mvs_eval/assets/shape/*.obj'))
if len(gt_mesh_files) != 6:
    bbox_diagonal = 0.24
else:
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


est_shape_mat(
    test_dataset, 
    out_dir,
    device,
    num_views_for_rfs = num_views_for_rfs,
    numdepth=numdepth,
    bbox_diagonal=bbox_diagonal,
    wo_sfs=args.wo_sfs,
    occlusion_handling=(not args.wo_occlusion_detection),
    wo_coarse=args.wo_coarse,
    wo_fine=args.wo_fine
)