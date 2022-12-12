import sys
import os

import torch

import numpy as np
import trimesh

from core.dataset import DrexelMultiNatGeomDataset
from est_shape_mat import est_shape_mat

import argparse

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numdepth = 192
num_views_for_rfs = 3

dataset_dir = os.environ['HOME']+'/data/nlmvs-real'
exp_name = 'nlmvsr'

list_illum_name = sorted(os.listdir(dataset_dir))
list_obj_name = ['ball', 'bunny', 'horse', 'planck', 'shell']
argv = sys.argv
if len(argv) == 2:
    list_available_set = []
    for illum_name in list_illum_name:
        for obj_name in sorted([obj_name for obj_name in os.listdir(dataset_dir+'/'+illum_name) if os.path.isdir(dataset_dir+'/'+illum_name+'/'+obj_name)]):
            list_available_set.append([illum_name, obj_name])
    idx_data = int(argv[1])    
    illum_name, obj_name = list_available_set[idx_data]
#elif len(argv) != 3:
#    print('usage: python diff_rendering_multinatgeom.py <illumination name> <object name>')
#    exit()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('illum_name', type=str)
    parser.add_argument('obj_name', type=str)
    parser.add_argument('--dataset-path', default=os.environ['HOME']+'/data/nlmvs-real', type=str)
    parser.add_argument('--exp-name', default='nlmvsr', type=str)
    args = parser.parse_args()

    illum_name = args.illum_name
    obj_name = args.obj_name

    dataset_dir = args.dataset_path
    exp_name = args.exp_name


out_dir = './run/est_shape_mat/'+exp_name+'/'+illum_name+'/'+obj_name
os.makedirs(out_dir, exist_ok=True)

test_dataset = DrexelMultiNatGeomDataset(dataset_dir, illum_name, obj_name)

print('illum name:', illum_name)
print('object name:', obj_name)
print('out_dir:', out_dir)


# compute bbox diagonal of gt geometry
gt_mesh_file = dataset_dir+'/'+illum_name+'/'+obj_name+'/mesh_aligned.ply'

mesh = trimesh.load(gt_mesh_file)
verts = np.asarray(mesh.vertices)

bbox_min = np.min(verts, axis=0)
bbox_max = np.max(verts, axis=0)
bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
print('diagonal of gt mesh:', bbox_diagonal)

est_shape_mat(
    test_dataset, 
    out_dir,
    device,
    num_views_for_rfs=num_views_for_rfs,
    numdepth=numdepth,
    bbox_diagonal=bbox_diagonal
)