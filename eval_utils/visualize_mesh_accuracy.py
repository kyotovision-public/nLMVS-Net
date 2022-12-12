import numpy as np
import cv2
import open3d as o3d
import trimesh

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', help='ply or obj file to load', type=str)
    parser.add_argument('target_mesh_file', help='ply or obj file to load', type=str)
    parser.add_argument('-om', '--out-mesh-file', help='mask file to save', type=str)
    args = parser.parse_args()

    def e2c(e, emax=1):
        e = (255 * np.clip(e,0,emax) / emax).astype(np.uint8)
        c = cv2.applyColorMap(e[None,:],cv2.COLORMAP_JET)[0][:,::-1]
        return c / 255.0
    gt_mesh_tri = trimesh.load(args.target_mesh_file)
    gt_mesh = o3d.io.read_triangle_mesh(args.target_mesh_file)
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    gt_verts = np.asarray(gt_mesh.vertices)
    verts = np.asarray(mesh.vertices)
    bbox_min = np.min(gt_verts, axis=0)
    bbox_max = np.max(gt_verts, axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
    verts_accuracy = gt_mesh_tri.nearest.on_surface(verts)[1] / bbox_diagonal
    mesh.vertex_colors = o3d.utility.Vector3dVector(e2c(100*verts_accuracy,3))
    o3d.io.write_triangle_mesh(args.out_mesh_file, mesh)