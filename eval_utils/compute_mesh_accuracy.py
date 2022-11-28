import numpy as np
import cv2
import open3d as o3d
import trimesh

import numba
from numba import cuda
import math

max_block_dim = 512

@cuda.jit(device=True)
def add_device(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

@cuda.jit(device=True)
def sub_device(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

@cuda.jit(device=True)
def dot_device(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] +v1[2] * v2[2]

@cuda.jit(device=True)
def scale_vector_device(scale, v):
    return scale * v[0], scale * v[1], scale * v[2]

@cuda.jit(device=True)
def cross_device(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2
    x = y1*z2 - y2*z1
    y = z1*x2 - z2*x1
    z = x1*y2 - x2*y1
    return x,y,z

@cuda.jit(device=True)
def normalize_device(v):
    x,y,z = v
    l = math.sqrt(x*x+y*y+z*z)+1e-12
    return x/l, y/l, z/l

# pts: Np*3
# face_verts: Nf*3*3
@cuda.jit
def compute_point_mesh_distance_kernel(pts, face_verts, result):
    idx_pt = cuda.blockIdx.x
    pt = pts[idx_pt]

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    min_dist_tmp = cuda.shared.array(shape=(max_block_dim,), dtype=numba.float32)
    min_dist_tmp[idx_thread] = 100000.0
    idx_face = idx_thread
    

    while idx_face < len(face_verts):
        p0 = face_verts[idx_face, 0]
        p1 = face_verts[idx_face, 1]
        p2 = face_verts[idx_face, 2]

        diff0, diff1, diff2 = sub_device(pt, p0), sub_device(pt, p1), sub_device(pt, p2)
        dist = math.sqrt(min(dot_device(diff0, diff0), dot_device(diff1, diff1), dot_device(diff2, diff2)))

        a = sub_device(p1, p0)
        b = sub_device(p2, p0)
        c = normalize_device(cross_device(a, b))

        u = dot_device(sub_device(pt, p0), c)
        pt_ = sub_device(sub_device(pt, p0), scale_vector_device(u, c))

        aDa = dot_device(a,a)
        bDb = dot_device(b,b)
        aDb = dot_device(a,b)

        xdDa = dot_device(pt_, a)
        xdDb = dot_device(pt_, b)

        det = bDb * aDa - aDb * aDb
        if abs(det) > 1e-20:
            s = (bDb * xdDa - aDb * xdDb) / det
            t = (-aDb * xdDa + aDa * xdDb) / det

            if (s >= 0) and (s <= 1) and (t >= 0) and (t <= 1):
                dist = min(abs(u), dist)

        if dist < min_dist_tmp[idx_thread]: 
            min_dist_tmp[idx_thread] = dist

        idx_face += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        min_dist = min_dist_tmp[0]
        for j in range(1,blockdim):
            if min_dist_tmp[j] < min_dist:
                min_dist = min_dist_tmp[j]
        result[idx_pt] = min_dist

def compute_mesh_rmse(mesh_file, target_mesh_file):
    gt_mesh_tri = trimesh.load(target_mesh_file)
    gt_mesh = o3d.io.read_triangle_mesh(target_mesh_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    gt_verts = np.asarray(gt_mesh.vertices)
    gt_faces = np.asarray(gt_mesh.triangles)
    gt_face_verts = np.stack([np.stack([gt_verts[idx] for idx in f]) for f in gt_faces])
    verts = np.asarray(mesh.vertices)
    bbox_min = np.min(gt_verts, axis=0)
    bbox_max = np.max(gt_verts, axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    pcd = np.asarray(mesh.sample_points_uniformly(100000, seed=16).points)
    
    if False:
        distances = np.zeros_like(pcd[:,0])
        compute_point_mesh_distance_kernel[(len(pcd),), (max_block_dim,)](pcd, gt_face_verts, distances)
        distances /= bbox_diagonal
    else:
        (closest_points, distances, triangle_id) = gt_mesh_tri.nearest.on_surface(pcd)
        distances /= bbox_diagonal

    rmse = np.sqrt(np.mean(distances**2))
    return rmse

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', help='ply or obj file to load', type=str)
    parser.add_argument('target_mesh_file', help='ply or obj file to load', type=str)
    args = parser.parse_args()

    rmse = compute_mesh_rmse(args.mesh_file, args.target_mesh_file)
    print('RMSE:', 100 * rmse, '%')