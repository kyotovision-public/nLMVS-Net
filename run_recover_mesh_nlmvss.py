import sys
import os
from glob import glob

import numpy as np
import open3d as o3d

import argparse

from recover_mesh import recover_mesh


parser = argparse.ArgumentParser()
parser.add_argument('object_name', help='name of the target object (e.g., 00152)', type=str)
parser.add_argument('--wo-sfs', action='store_true')
parser.add_argument('--exp-name', default='nlmvss', type=str)
parser.add_argument('--gt-mesh-path', default=None, type=str)

args = parser.parse_args()

object_name = args.object_name

out_dir = './run/mesh'
result_dir = './run/est_shape_mat'
if args.wo_sfs:
    out_dir = out_dir+'_wo_sfs'
    result_dir = result_dir+'_wo_sfs'
out_dir = out_dir+'/'+args.exp_name+'/'+object_name
result_dir = result_dir+'/'+args.exp_name+'/'+object_name+'/final_results'
os.makedirs(out_dir, exist_ok=True)
print('result_dir:', result_dir)
print('out_dir:', out_dir)

# create unnormalized gt mesh
if args.gt_mesh_path is None:
    gt_mesh_file = None
else:
    gt_mesh_files = sorted(glob(args.gt_mesh_path+'/*.obj'))
    object_id = int(object_name)
    gt_mesh_file = gt_mesh_files[(object_id // 6) % len(gt_mesh_files)]

    mesh = o3d.io.read_triangle_mesh(gt_mesh_file)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
    camera_min_distance = 1.0
    camera_tangent = 1.0 / (2.0 / 1200 * 5000.0)
    scale = 0.5 * bbox_diagonal / camera_tangent / camera_min_distance

    verts -= bbox_center
    verts /= scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d.io.write_triangle_mesh(out_dir+'/mesh_gt.ply', mesh)

    gt_mesh_file = out_dir+'/mesh_gt.ply'

recover_mesh(
    result_dir, 
    out_dir,
    gt_mesh_file=gt_mesh_file,
    render_results=True
)