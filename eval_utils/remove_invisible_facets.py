import numpy as np
import numba
from numba import cuda
import open3d as o3d
import cv2

@cuda.jit(device=True)
def dot_device(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] +v1[2] * v2[2]

@cuda.jit(device=True)
def cross_device(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2
    x = y1*z2 - y2*z1
    y = z1*x2 - z2*x1
    z = x1*y2 - x2*y1
    return x,y,z

# img:   H*W*C
# depth: H*W
# mask: H*W
# verts: N*3*2
# verts_color: N*3*C
# verts_depth: N*3
@cuda.jit
def render_face_indices_kernel(idx_map, depth, mask, verts, verts_depth):
    H,W = idx_map.shape
    N = len(verts)

    idx_y = cuda.blockIdx.x
    idx_x = cuda.blockIdx.y

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    x = float(idx_x)
    y = float(idx_y)
    d = (x,y,1.0)

    max_block_dim = 256
    color_buffer = cuda.shared.array(shape=(max_block_dim,), dtype=numba.int32)
    color_buffer[idx_thread] = -1
    z_buffer = cuda.shared.array(shape=(max_block_dim), dtype=numba.float32)
    z_buffer[idx_thread] = 1e6
    mask_buffer = cuda.shared.array(shape=(max_block_dim), dtype=numba.float32)
    mask_buffer[idx_thread] = 0.0

    idx_face = idx_thread
    while (mask[idx_y,idx_x] > 0.0) and (idx_face < N):
        za,zb,zc = verts_depth[idx_face]
        min_z = min(za,zb,zc)
        if z_buffer[idx_thread] < min_z:
            break

        a,b,c = verts[idx_face]
        min_x = min(a[0], b[0], c[0])
        min_y = min(a[1], b[1], c[1])
        max_x = max(a[0], b[0], c[0])
        max_y = max(a[1], b[1], c[1])
        mgn = 2
        if (x >= (min_x - mgn)) and (y >= (min_y - mgn)) and (x <= (max_x + mgn)) and (y <= (max_y + mgn)):
            r = (-a[0]*za, -a[1]*za, -za)
            e1 = (b[0]*zb - a[0]*za, b[1]*zb -a[1]*za, zb - za)
            e2 = (c[0]*zc - a[0]*za, c[1]*zc -a[1]*za, zc - za)

            denom = dot_device(cross_device(d, e2), e1)
            if abs(denom) > 1e-12:
                t = dot_device(cross_device(r, e1), e2) / denom
                u = dot_device(cross_device(d, e2), r) / denom
                v = dot_device(cross_device(r, e1), d) / denom
                w = 1 - u - v

                eps = 1e-3
                if (t >= 0) and (u >= (0.0 - eps)) and (u <= (1.0 + eps)) and (v >= (0.0 - eps)) and (v <= (1.0 + eps)) and (w >= (0.0 - eps)) and (w <= (1.0 + eps)):
                    # compute depth
                    z = max(min_z, t)
                    if z < z_buffer[idx_thread]:
                        mask_buffer[idx_thread] = 1.0
                        z_buffer[idx_thread] = z

                        # compute color
                        color_buffer[idx_thread] = idx_face


        idx_face += blockdim

    cuda.syncthreads()
    if idx_thread == 0:
        for idx in range(1,blockdim):
            if z_buffer[idx] < z_buffer[0]:
                color_buffer[0] = color_buffer[idx] 
                mask_buffer[0] = mask_buffer[idx]
                z_buffer[0] = z_buffer[idx]

        # write results
        idx_map[idx_y, idx_x] = color_buffer[0]
        mask[idx_y, idx_x] = mask[idx_y, idx_x] * mask_buffer[0]
        depth[idx_y, idx_x] = z_buffer[0] * mask[idx_y, idx_x]

# verts: N*3*2
# verts_color: N*3*C
# verts_depth: N*3
# ret:  img(H*W*C), depth(H*W), mask(H*W)
def render_face_indices(verts, verts_depth, image_size, ref_mask=None):
    W,H = image_size
    N = len(verts)
    C = 3#verts_color.shape[2]

    # sort according to min depth
    indices = np.argsort(np.min(verts_depth, axis=1), axis=0)
    verts = verts[indices]
    #verts_color = verts_color[indices]
    verts_depth = verts_depth[indices]

    # precompute mask
    mask = np.zeros((H,W), dtype=np.float32)
    for pts in verts:
        cv2.fillConvexPoly(mask, pts.astype(np.int32), (1.0, 1.0, 1.0))
    if ref_mask is not None:
        mask *= ref_mask

    verts = verts.astype(np.float32)
    #verts_color = verts_color.astype(np.float32)
    verts_depth = verts_depth.astype(np.float32)
    mask = mask.astype(np.float32)

    idx_map = np.zeros((H,W), dtype=np.int32)
    depth_map = np.zeros((H,W), dtype=np.float32)

    render_face_indices_kernel[(H,W), (256)](idx_map, depth_map, mask, verts, verts_depth)

    idx_map_original = indices[idx_map]
    idx_map_original[idx_map == -1] = -1
    return idx_map_original, depth_map, mask

def eval_mesh_visibility(verts, faces, proj_matrix, imsize, ref_mask = None):
    W,H = imsize
    face_verts3d = np.stack([np.stack([verts[idx] for idx in f]) for f in faces])
    face_verts2d_h = np.array([(proj_matrix[:3,:3]@verts3d.transpose(1,0) + proj_matrix[:3,3:4]).transpose(1,0) for verts3d in face_verts3d]) # [N,3,3]
    face_verts_depth = np.array([v[:,2] for v in face_verts2d_h]) # [N,3]
    face_verts2d = np.array([np.stack((v[:,0] / v[:,2], v[:,1] / v[:,2]), axis=1) for v in face_verts2d_h]) # [N,3,2]

    diagonal = np.linalg.norm(np.max(verts, axis=0) - np.min(verts, axis=0))

    max_edge_len = np.sqrt(np.max(np.stack([np.sum((face_verts2d[:,1] - face_verts2d[:,0])**2, axis=-1), np.sum((face_verts2d[:,2] - face_verts2d[:,1])**2, axis=-1), np.sum((face_verts2d[:,0] - face_verts2d[:,2])**2, axis=-1)])))
    print('maximum edge length:', max_edge_len)
    ratio = int(np.clip(100/max_edge_len, 1, 5))
    ref_mask = cv2.resize(ref_mask, (ratio*W,ratio*H), interpolation=cv2.INTER_NEAREST)
    idx_map, depth_map, rendered_mask = render_face_indices(ratio * face_verts2d, face_verts_depth, (ratio*W,ratio*H), ref_mask)

    visible_face_indices = np.unique(idx_map[idx_map >= 0])
    visible_faces = faces[visible_face_indices]

    # 以下の条件を満たす場合もvisibleと判定する
    # - 3つの頂点が全てvisible
    # - 見かけの大きさが非常に小さい(e.g., 2*面積/長辺の長さ<1pix以下とか?) -- not implemented yet
    # - 重心位置における深度が深度マップと整合(エラーがdiagonalの0.1%以下?)
    #print(visible_indices)
    #print(sorted(idx_map[idx_map >= 0]))
    visible_vertex_indices = np.unique(visible_faces.reshape(-1))
    invisible_face_indices = np.setdiff1d(np.arange(len(faces)), visible_face_indices)
    invisible_faces = faces[invisible_face_indices]
    verts_visibility = np.zeros_like(verts[:,0], dtype=bool)
    verts_visibility[visible_vertex_indices] = True
    invisible_face_verts_visibility = np.all(verts_visibility[invisible_faces], axis=1)
    invisible_face_center = np.mean(face_verts2d[invisible_face_indices], axis=1) # N*2
    invisible_face_center_depth = np.mean(face_verts_depth[invisible_face_indices], axis=1) # N
    chunk_size = 1024 * 4
    invisible_face_center_depth_sampled_chunks = []
    for idx_chunk in range((len(invisible_face_center) - 1) // chunk_size + 1):
        invisible_face_center_depth_sampled_chunk = cv2.remap(
            depth_map.astype(np.float32), 
            ratio * invisible_face_center[idx_chunk*chunk_size:(idx_chunk+1)*chunk_size,0:1].astype(np.float32), 
            ratio * invisible_face_center[idx_chunk*chunk_size:(idx_chunk+1)*chunk_size,1:2].astype(np.float32), 
            interpolation=cv2.INTER_LINEAR
        )[:,0]
        invisible_face_center_depth_sampled_chunks.append(invisible_face_center_depth_sampled_chunk)
    invisible_face_center_depth_sampled = np.concatenate(invisible_face_center_depth_sampled_chunks, axis=0)
    face_center_depth_error = np.abs(invisible_face_center_depth_sampled - invisible_face_center_depth) / diagonal
    invisible_face_center_visibility = face_center_depth_error < 0.001

    visible_face_indices = np.unique(np.concatenate([
        visible_face_indices,
        invisible_face_indices[invisible_face_verts_visibility * invisible_face_center_visibility]
    ]))
    return visible_face_indices


if True:
    import argparse
    from glob import glob
    # Example: python render_ply.py 3D/pig.ply 3476 5208 -i indoor/checkerboard/intrinsics.npy -e indoor/pig/view-01_extrinsics.npy
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='e.g., ./run/oriented_point/mvs_eval/00152', type=str)
    parser.add_argument('-n', '--num-neighbors', help='number of neghboring views', type=int, default=4)
    args = parser.parse_args()

    mesh_file = args.result_dir+'/mesh_poisson.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()

    print('number of faces:', len(faces))
    print('number of vertices:', len(verts))

    view_dirs = sorted(glob(args.result_dir+'/view-??'))
    visibility_per_view = []
    face_num_visible = np.zeros_like(faces[:,0])
    for view_dir in view_dirs:
        intrinsic_file = view_dir+'/intrinsics.npy'
        extrinsic_file = view_dir+'/extrinsics.npy'
        ref_mask_file = view_dir+'/ref_mask.png'
        K = np.load(intrinsic_file)
        T = np.load(extrinsic_file)
        K = np.concatenate([np.concatenate([K, np.zeros_like(K[:,:1])], axis=1), np.array([[0,0,0,1]])], axis=0)
        P = K@T

        ref_mask = cv2.imread(ref_mask_file,0)
        ref_mask = ref_mask.astype(np.float32) / 255.0
        H,W = ref_mask.shape

        visible_face_indices = eval_mesh_visibility(verts, faces, P, (W,H), ref_mask)
        visible_faces = faces[visible_face_indices]

        mesh.triangles = o3d.utility.Vector3iVector(visible_faces)
        o3d.io.write_triangle_mesh(view_dir+'/visible_faces.ply', mesh)
        print('created', view_dir+'/visible_faces.ply')

        face_num_visible[visible_face_indices] += 1
        vis = np.zeros((len(faces),), dtype=bool)
        vis[visible_face_indices] = True
        visibility_per_view.append(vis)

    visible_faces = faces[face_num_visible > 0]
    mesh.triangles = o3d.utility.Vector3iVector(visible_faces)
    o3d.io.write_triangle_mesh(args.result_dir+'/mesh.ply', mesh)
    print('created', args.result_dir+'/mesh.ply')

    overall_visibility = np.zeros((len(faces),), dtype=bool)
    for view_idx, view_dir in enumerate(view_dirs):
        vis = np.ones((len(faces),), dtype=bool)
        neighboring_views = np.loadtxt(view_dir+'/views.txt', dtype=int)[:1+args.num_neighbors]
        for j in neighboring_views:
            vis *= visibility_per_view[j]
        mesh.triangles = o3d.utility.Vector3iVector(faces[vis])
        o3d.io.write_triangle_mesh(view_dir+'/visible_faces_'+str(1+args.num_neighbors)+'.ply', mesh)
        print('created', view_dir+'/visible_faces_'+str(1+args.num_neighbors)+'.ply')

        overall_visibility[vis] = True
        
        

    visible_faces = faces[overall_visibility]

    
    mesh.triangles = o3d.utility.Vector3iVector(visible_faces)
    o3d.io.write_triangle_mesh(args.result_dir+'/mesh_visible.ply', mesh)
    print('created', args.result_dir+'/mesh_visible.ply')