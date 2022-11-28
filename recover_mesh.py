import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.sfs_utils import *
from eval_utils.compute_mesh_accuracy import compute_mesh_rmse

import numpy as np

from tqdm import tqdm
from glob import glob
import subprocess
import json

torch.manual_seed(0)
device = torch.device("cpu")

# save results
def save_oriented_points_as_ply(
    verts, 
    normals, 
    out_file
):
    verts = verts.cpu().numpy()
    normals = normals.cpu().numpy()

    properties = np.concatenate([verts, normals], axis=1)
    s = '\n'.join(['ply',
        'format ascii 1.0',
        'comment VCGLIB generated',
        'element vertex '+str(len(properties)),
        'property float x',
        'property float y',
        'property float z',
        'property float nx',
        'property float ny',
        'property float nz',
        'end_header\n'])
    s = s + ''.join([' '.join([str(v) for v in p])+'\n' for p in properties])
    with open(out_file, 'w') as f:
        f.write(s)

def create_views(out_file):
    text = '1200 1200\n'
    text += '24000 0.000000 600.00000 0.000000 24000 600.0000 0.000000 0.000000 1.000000'
    for idx, phi in enumerate(np.arange(0,2.0*np.pi,2.0*np.pi/360)):
        up = np.array([0.,1.,0.])
        look = np.array([np.cos(phi), 0, np.sin(phi)])
        right = np.cross(look,up)
        rot = np.stack([right,-up,look], axis=0)
        rvec = cv2.Rodrigues(rot)[0][:,0]
        tvec = np.array([0.,0.,2.0])
        text += '\n'+str(idx).zfill(8)+' '+' '.join([str(val) for val in tvec] + [str(val) for val in rvec])

    with open(out_file, 'w') as f:
        f.write(text)

def recover_mesh(
    result_dir, 
    out_dir,
    gt_mesh_file = None,
    compute_per_view_errors = False,
    compute_mesh_errors = False,
    render_results = False,
    num_neighbors=4
):
    views_dirs = sorted(glob(result_dir+'/view-??'))
    bar = tqdm(views_dirs)
    bar.set_description('Point cloud reconstruction')
    list_verts = []
    list_verts_normals = []
    for idx_view, view_dir in enumerate(bar):
        est_depth = torch.from_numpy(np.load(view_dir+'/est_depth.npy')[None,None,:,:]).to(device)
        est_normal = torch.from_numpy(np.load(view_dir+'/est_normal.npy').transpose(2,0,1)[None,:,:,:]).to(device)
        est_mask = torch.from_numpy(cv2.imread(view_dir+'/vis_mask.png',0).astype(np.float32) / 255.0).to(device)[None,None,:,:]

        intrinsic_matrix = torch.from_numpy(np.load(view_dir+'/intrinsics.npy')[None]).to(device)
        extrinsic_matrix = torch.from_numpy(np.load(view_dir+'/extrinsics.npy')[None]).to(device)
        proj_matrix = extrinsic_matrix.clone()
        proj_matrix[:,:3,:4] = intrinsic_matrix @ extrinsic_matrix[:,:3,:4]

        # create oriented points
        v,u = torch.meshgrid(torch.arange(est_depth.size(2)), torch.arange(est_depth.size(3)))
        u = (u.float() + 0.5).to(device)
        v = (v.float() + 0.5).to(device)
        d = est_depth[:,0]
        ones = torch.ones_like(d, dtype=d.dtype, device=device)
        m = torch.stack([u*d,v*d,d,ones], dim=1) # BS,4,H,W
        m = m.transpose(1,2).transpose(2,3).view(-1,4) # H*W,4
        pts_pos = (torch.inverse(proj_matrix) @ m[:,:,None])[:,:3,0] # [H*W,3]
        n = est_normal.transpose(1,2).transpose(2,3).view(-1,3) # H*W,3
        n = n * torch.tensor([1.0, -1.0, -1.0], dtype=n.dtype, device=device)[None,:]
        pts_normal = (torch.inverse(extrinsic_matrix[:,:3,:3]) @ n[:,:,None])[:,:3,0] # H*W,3
        
        # filtering according to the depth probability
        pts_mask = (est_mask > 0.5).view(-1)
        verts = pts_pos[pts_mask]
        verts_normals = pts_normal[pts_mask]

        list_verts.append(verts)
        list_verts_normals.append(verts_normals)

        # copy results for visibility test
        per_view_out_dir = out_dir+'/view-'+str(idx_view+1).zfill(2)
        os.makedirs(per_view_out_dir, exist_ok=True)
        subprocess.run(['cp', view_dir+'/ref_mask.png', per_view_out_dir+'/ref_mask.png'])
        subprocess.run(['cp', view_dir+'/intrinsics.npy', per_view_out_dir+'/intrinsics.npy'])
        subprocess.run(['cp', view_dir+'/extrinsics.npy', per_view_out_dir+'/extrinsics.npy'])
        subprocess.run(['cp', view_dir+'/views.txt', per_view_out_dir+'/views.txt'])

        save_oriented_points_as_ply(
            verts, 
            verts_normals, 
            per_view_out_dir+'/pcd.ply'
        )
        if compute_per_view_errors and (not (gt_mesh_file is None)):
            subprocess.run([
                'python', 'eval_utils/visualize_mesh_accuracy.py', 
                per_view_out_dir+'/pcd.ply',
                gt_mesh_file,
                '-om', per_view_out_dir+'/pcd_error.ply',
        ])

    out_file = out_dir+'/merged_points.ply'
    merged_verts = torch.cat(list_verts, dim=0)
    merged_verts_normals = torch.cat(list_verts_normals, dim=0)
    save_oriented_points_as_ply(merged_verts, merged_verts_normals, out_file)

    # remeshing
    in_file = out_dir+'/merged_points.ply'
    out_file = out_dir+'/mesh_poisson.ply'
    subprocess.run([
        'meshlabserver', 
        '-i', in_file, 
        '-o', out_file, 
        '-m', 'sa', 
        '-s', './data/remeshing.mlx'
    ])

    subprocess.run([
        'python',
        'eval_utils/remove_invisible_facets.py',
        out_dir,
        '-n', str(num_neighbors)
    ])

    if compute_mesh_errors and (not (gt_mesh_file) is None):
        # compute mesh accuracy (Hausdorf distance)
        subprocess.run([
            'python', 'eval_utils/visualize_mesh_accuracy.py', 
            out_dir+'/mesh.ply',
            gt_mesh_file,
            '-om', out_dir+'/mesh_error.ply',
        ])

        subprocess.run([
            'python', 'eval_utils/visualize_mesh_accuracy.py', 
            out_dir+'/mesh_visible.ply',
            gt_mesh_file,
            '-om', out_dir+'/mesh_visible_error.ply',
        ])

        rmse_full = compute_mesh_rmse(out_dir+'/mesh.ply', gt_mesh_file)
        rmse_visible = compute_mesh_rmse(out_dir+'/mesh_visible.ply', gt_mesh_file)
        err_dict = {
            'rmse_full': rmse_full,
            'rmse_visible': rmse_visible,
        }
        with open(out_dir+'/errors.json', 'w') as f:
            json.dump(err_dict, f, ensure_ascii=True)

    # postprocessing for rendering
    subprocess.run([
        'meshlabserver', 
        '-i', out_dir+'/mesh.ply', 
        '-o', out_dir+'/mesh.obj', 
        '-m', 'vn',
        '-s', './data/close_holes.mlx',
    ])
    if not os.path.exists(out_dir+'/mesh.obj'):
        subprocess.run([
            'meshlabserver', 
            '-i', out_dir+'/mesh.ply', 
            '-o', out_dir+'/mesh.obj', 
            '-m', 'vn',
        ])

    if not render_results:
        return

    create_views(out_dir+'/views_for_rendering.txt')

    subprocess.run(['ln', '-s', os.path.abspath(result_dir+'/brdf.binary'), out_dir+'/brdf.binary'])

    # render results
    views_dirs = sorted(glob(out_dir+'/view-??'))
    for view_dir in views_dirs:
        # render estimation result
        subprocess.run([
            'python', 'eval_utils/render_geometry_mgl.py', 
            '-i', view_dir+'/intrinsics.npy',
            '-e', view_dir+'/extrinsics.npy',
            '-n', view_dir+'/mesh_normal.npy',
            '-d', view_dir+'/mesh_depth.npy',
            '-m', view_dir+'/mesh_mask.png',
            '--up-sampling', '4',
            out_dir+'/mesh.obj',
            '128', '128',
        ])
        cv2.imwrite(view_dir+'/mesh_shading.png', (255*(np.clip(np.abs(np.load(view_dir+'/mesh_normal.npy')[:,:,2]), 0, 1)**(1/2.2))).astype(np.uint8))
        print('created', view_dir+'/mesh_shading.png')

        img_shading = (255*(np.clip(0.8*np.load(view_dir+'/mesh_normal.npy')[:,:,2], 0, 1)**(1/2.2))).astype(np.uint8)
        mesh_mask = cv2.imread(view_dir+'/mesh_mask.png', 0)
        cv2.imwrite(view_dir+'/mesh_shading_tran.png', np.concatenate([
            img_shading[:,:,None], img_shading[:,:,None], img_shading[:,:,None], mesh_mask[:,:,None]
        ], axis=-1))
        print('created', view_dir+'/mesh_shading_tran.png')

        ## render gt mesh
        subprocess.run([
            'python', 'eval_utils/render_geometry_mgl.py', 
            '-i', view_dir+'/intrinsics.npy',
            '-e', view_dir+'/extrinsics.npy',
            '-n', view_dir+'/gt_mesh_normal.npy',
            '-d', view_dir+'/gt_mesh_depth.npy',
            '-m', view_dir+'/gt_mesh_mask.png',
            '--up-sampling', '4',
            out_dir+'/mesh_gt.ply',
            '128', '128',
        ])
        cv2.imwrite(view_dir+'/gt_mesh_shading.png', (255*(np.clip(np.abs(np.load(view_dir+'/gt_mesh_normal.npy')[:,:,2]), 0, 1)**(1/2.2))).astype(np.uint8))
        print('created', view_dir+'/gt_mesh_shading.png')

        img_shading = (255*(np.clip(0.8*np.load(view_dir+'/gt_mesh_normal.npy')[:,:,2], 0, 1)**(1/2.2))).astype(np.uint8)
        mesh_mask = cv2.imread(view_dir+'/gt_mesh_mask.png', 0)
        cv2.imwrite(view_dir+'/gt_mesh_shading_tran.png', np.concatenate([
            img_shading[:,:,None], img_shading[:,:,None], img_shading[:,:,None], mesh_mask[:,:,None]
        ], axis=-1))
        print('created', view_dir+'/gt_mesh_shading_tran.png')