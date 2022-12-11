import sys
import os
import subprocess

import numpy as np
import cv2

from glob import glob
import argparse

from recover_mesh import recover_mesh


dataset_dir = os.environ['HOME']+'/data/nlmvs-real'

list_illum_name = sorted(os.listdir(dataset_dir))
list_obj_name = ['ball', 'bunny', 'horse', 'planck', 'shell']
argv = sys.argv
if len(argv) == 3:
    list_available_set = []
    for illum_name in list_illum_name:
        for obj_name in sorted([obj_name for obj_name in os.listdir(dataset_dir+'/'+illum_name) if os.path.isdir(dataset_dir+'/'+illum_name+'/'+obj_name)]):
            list_available_set.append([illum_name, obj_name])
    idx_data = int(argv[1])
    view_index = int(argv[2])
    illum_name, obj_name = list_available_set[idx_data]
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('illum_name', type=str)
    parser.add_argument('obj_name', type=str)
    parser.add_argument('view_index', help='index of the reference view', type=int)
    parser.add_argument('--dataset-path', default=os.environ['HOME']+'/data/nlmvs-real', type=str)
    parser.add_argument('--exp-name', default='nlmvsr', type=str)
    args = parser.parse_args()

    illum_name = args.illum_name
    obj_name = args.obj_name
    view_index = args.view_index

    dataset_dir = args.dataset_path
    exp_name = args.exp_name

    list_illum_name = sorted(os.listdir(dataset_dir))

    if not (illum_name in list_illum_name):
        print('illumination name must be one of ', list_illum_name)
        exit()
    obj_name = str(argv[2])
    if not (obj_name in list_obj_name):
        print('object name must be one of ', list_obj_name)
        exit()

out_dir = './run/mesh_per_view/'+exp_name+'/'+illum_name+'/'+obj_name+'/'+str(view_index).zfill(3)
result_dir = './run/est_shape_mat_per_view/'+exp_name+'/'+illum_name+'/'+obj_name+'/'+str(view_index).zfill(3)+'/final_results'

os.makedirs(out_dir, exist_ok=True)
print('result_dir:', result_dir)
print('out_dir:', out_dir)

# create unnormalized gt mesh
gt_mesh_file = dataset_dir+'/'+illum_name+'/'+obj_name+'/mesh_aligned.ply'
subprocess.run(['cp', gt_mesh_file, out_dir+'/mesh_gt.ply'])

recover_mesh(
    result_dir, 
    out_dir,
    gt_mesh_file=out_dir+'/mesh_gt.ply',
    render_results=True,
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
os.makedirs(out_dir+'/novel_view_rendering', exist_ok=True)
num_novel_views = len(glob('./run/est_shape_mat/'+exp_name+'/'+illum_name+'/'+obj_name+'/final_results/view-??'))
for view_id in range(num_novel_views):
    view_dir = out_dir+'/novel_view_rendering/view-'+str(1+view_id).zfill(2)
    os.makedirs(view_dir, exist_ok=True)
    intrinsic_file = './run/est_shape_mat/nlmvsr/'+illum_name+'/'+obj_name+'/final_results/view-'+str(1+view_id).zfill(2)+'/intrinsics.npy'
    extrinsic_file = './run/est_shape_mat/nlmvsr/'+illum_name+'/'+obj_name+'/final_results/view-'+str(1+view_id).zfill(2)+'/extrinsics.npy'

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