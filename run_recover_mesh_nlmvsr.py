import subprocess
import sys
import os
import cv2
from glob import glob

import argparse

from recover_mesh import recover_mesh

dataset_dir = os.environ['HOME']+'/data/nlmvs-real'

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

    list_illum_name = sorted(os.listdir(dataset_dir))

    if not (illum_name in list_illum_name):
        print('illumination name must be one of ', list_illum_name)
        exit()
    obj_name = str(argv[2])
    if not (obj_name in list_obj_name):
        print('object name must be one of ', list_obj_name)
        exit()

result_dir = './run/est_shape_mat/'+exp_name+'/'+illum_name+'/'+obj_name+'/final_results'
out_dir = './run/mesh/'+exp_name+'/'+illum_name+'/'+obj_name
os.makedirs(out_dir, exist_ok=True)
print('out_dir:', out_dir)

# create unnormalized gt mesh
gt_mesh_file = dataset_dir+'/'+illum_name+'/'+obj_name+'/mesh_aligned.ply'
subprocess.run(['cp', gt_mesh_file, out_dir+'/mesh_gt.ply'])

recover_mesh(
    result_dir, 
    out_dir,
    gt_mesh_file=out_dir+'/mesh_gt.ply',
    render_results=True
)

# save small illumination map & ref img for appearance recovery
illum_map = cv2.imread(dataset_dir+'/'+illum_name+'/'+obj_name+'/illumination.exr', -1)
illum_map = cv2.resize(illum_map, dsize=(2048,1024), interpolation=cv2.INTER_AREA)
cv2.imwrite(out_dir+'/illumination.hdr', illum_map)

view_dirs = glob(out_dir+'/view-??')
for idx_view, view_dir in enumerate(view_dirs):
    subprocess.run(['ln', '-s', os.path.abspath(result_dir+'/view-'+str(1+idx_view).zfill(2)+'/img_0.png'), view_dir+'/img_0.png'])