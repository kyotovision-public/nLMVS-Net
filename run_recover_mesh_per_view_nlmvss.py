import sys
import os
from glob import glob
import subprocess

import numpy as np
import open3d as o3d
import cv2

import argparse

from recover_mesh import recover_mesh


parser = argparse.ArgumentParser()
parser.add_argument('object_name', help='name of the target object (e.g., 00152)', type=str)
parser.add_argument('view_index', help='index of the reference view', type=int)
parser.add_argument('--wo-sfs', action='store_true')
parser.add_argument('--exp-name', default='nlmvss', type=str)
args = parser.parse_args()

object_name = args.object_name
view_index = args.view_index

out_dir = './run/mesh_per_view'
result_dir = './run/est_shape_mat_per_view'
if args.wo_sfs:
    out_dir = out_dir+'_wo_sfs'
    result_dir = result_dir+'_wo_sfs'
out_dir = out_dir+'/'+args.exp_name+'/'+object_name+'/'+str(view_index).zfill(3)
result_dir = result_dir+'/'+args.exp_name+'/'+object_name+'/'+str(view_index).zfill(3)+'/final_results'
os.makedirs(out_dir, exist_ok=True)
print('result_dir:', result_dir)
print('out_dir:', out_dir)

# create unnormalized gt mesh
gt_mesh_files = sorted(glob(os.environ['HOME']+'/data/mvs_eval/assets/shape/*.obj'))
if len(gt_mesh_files) != 6:
    gt_mesh_file = None
else:
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
    compute_mesh_errors=False,
    num_neighbors=-1
)

subprocess.run([
    'meshlabserver', 
    '-i', out_dir+'/merged_points.ply', 
    '-o', out_dir+'/mesh_poisson.ply', 
    '-m', 'sa', 
    '-s', './data/remeshing_sharp.mlx'
])

subprocess.run([
    'meshlabserver', 
    '-i', out_dir+'/mesh_poisson.ply', 
    '-o', out_dir+'/mesh_poisson.ply', 
    '-m', 'sa', 
    '-s', './data/loop_subdivision.mlx'
])

subprocess.run([
    'python',
    'eval_utils/remove_invisible_facets.py',
    out_dir,
    '-n', str(-1)
])

subprocess.run([
    'meshlabserver', 
    '-i', out_dir+'/mesh.ply', 
    '-o', out_dir+'/mesh.ply', 
    '-m', 'sa', 
    '-s', './data/close_holes.mlx'
])

# render from other views
if not os.path.exists('./run/est_shape_mat/nlmvss/'+object_name+'/final_results'):
    exit()
os.makedirs(out_dir+'/novel_view_rendering', exist_ok=True)
for view_id in range(20):
    view_dir = out_dir+'/novel_view_rendering/view-'+str(1+view_id).zfill(2)
    os.makedirs(view_dir, exist_ok=True)
    intrinsic_file = './run/est_shape_mat/nlmvss/'+object_name+'/final_results/view-'+str(1+view_id).zfill(2)+'/intrinsics.npy'
    extrinsic_file = './run/est_shape_mat/nlmvss/'+object_name+'/final_results/view-'+str(1+view_id).zfill(2)+'/extrinsics.npy'

    subprocess.run([
        'python', 'eval_utils/render_geometry_mgl.py', 
        '-i', intrinsic_file,
        '-e', extrinsic_file,
        '-n', view_dir+'/mesh_normal.npy',
        '-d', view_dir+'/mesh_depth.npy',
        '-m', view_dir+'/mesh_mask.png',
        '--up-sampling', '4',
        out_dir+'/mesh.ply',
        '128', '128',
    ])
    img_shading = (255*(np.clip(0.8*np.load(view_dir+'/mesh_normal.npy')[:,:,2], 0, 1)**(1/2.2))).astype(np.uint8)
    mesh_mask = cv2.imread(view_dir+'/mesh_mask.png', 0)
    cv2.imwrite(view_dir+'/mesh_shading.png', np.concatenate([
        img_shading[:,:,None], img_shading[:,:,None], img_shading[:,:,None], mesh_mask[:,:,None]
    ], axis=-1))
    print('created', view_dir+'/mesh_shading.png')

    if not (gt_mesh_file is None):
        subprocess.run([
            'python', 'eval_utils/render_geometry_mgl.py', 
            '-i', intrinsic_file,
            '-e', extrinsic_file,
            '-n', view_dir+'/gt_mesh_normal.npy',
            '-d', view_dir+'/gt_mesh_depth.npy',
            '-m', view_dir+'/gt_mesh_mask.png',
            '--up-sampling', '4',
            out_dir+'/mesh_gt.ply',
            '128', '128',
        ])
        img_shading = (255*(np.clip(0.8*np.load(view_dir+'/gt_mesh_normal.npy')[:,:,2], 0, 1)**(1/2.2))).astype(np.uint8)
        mesh_mask = cv2.imread(view_dir+'/gt_mesh_mask.png', 0)
        cv2.imwrite(view_dir+'/gt_mesh_shading.png', np.concatenate([
            img_shading[:,:,None], img_shading[:,:,None], img_shading[:,:,None], mesh_mask[:,:,None]
        ], axis=-1))
        print('created', view_dir+'/gt_mesh_shading.png')