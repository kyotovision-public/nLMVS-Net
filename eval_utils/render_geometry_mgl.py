import moderngl
import numpy as np
import open3d as o3d

class MeshGeometryRenderer():
    def __init__(self, mesh_file, image_width, image_height, z_near=1e-3, z_far=1e4, extrinsics=None, intrinsics=None):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        verts_normals = np.asarray(mesh.vertex_normals)
        faces = np.asarray(mesh.triangles)
        vertices = []
        vertex_normals = []
        for face in faces:
            vertices.append(verts[face])
            vertex_normals.append(verts_normals[face])
        vertices = np.concatenate(vertices, axis=0)
        vertex_normals = np.concatenate(vertex_normals, axis=0)
        vertices = np.concatenate([vertices, vertex_normals], axis=1)

        # moderngl
        # reference: 
        # - https://github.com/moderngl/moderngl
        # - https://www.qoosky.io/techs/8f169ee13b
        self.ctx = moderngl.create_standalone_context()
        self.vbo = self.ctx.buffer(vertices.astype('float32').tobytes())

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_vert;
                in vec3 in_normal;
                out vec3 v_normal;

                uniform float z_near;
                uniform float z_far;
                uniform mat4 proj_matrix;
                uniform mat3 rot_matrix;
                uniform float image_width;
                uniform float image_height;

                void main() {
                    vec4 x_world = vec4(in_vert, 1.0);
                    vec4 x_opencv = proj_matrix * x_world;
                    mat4 M = mat4(
                        2 / image_width, 0, 0, 0,
                        0, 2 / image_height, 0, 0,
                        -1, -1, (z_far + z_near) / (z_far - z_near), 1,
                        0, 0, -2 * z_near * z_far / (z_far - z_near), 0
                    );
                    v_normal = vec3(1,-1,-1) * (rot_matrix * in_normal);
                    gl_Position = M * x_opencv;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                out vec4 f_color;
                void main() {
                    float norm = clamp(sqrt(
                        v_normal.x * v_normal.x +
                        v_normal.y * v_normal.y +
                        v_normal.z * v_normal.z
                    ), 0.001, 100000);
                    f_color = vec4(v_normal / norm, 0.0);
                }
            '''
        )

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.z_near = z_near
        self.z_far = z_far
        self.image_height = image_height
        self.image_width = image_width

        self.prog['z_near'].value = z_near
        self.prog['z_far'].value = z_far
        self.prog['image_width'].value = image_width
        self.prog['image_height'].value = image_height

        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_normal')

        self.fbo = self.ctx.simple_framebuffer((image_width, image_height), dtype='f4')
        self.fbo.use()

        self.ctx.enable(moderngl.DEPTH_TEST)

    def render(self, intrinsics=None, extrinsics=None):
        if not (intrinsics is None):
            self.intrinsics = intrinsics
        if not (extrinsics is None):
            self.extrinsics = extrinsics

        # compute camera parameters
        K = np.concatenate([np.concatenate([self.intrinsics, np.zeros((3,1))], axis=1), np.array([[0,0,0,1]])], axis=0)
        T = self.extrinsics
        rot_matrix = T[:3,:3]
        proj_matrix = K @ T
        self.prog['rot_matrix'].value = tuple(rot_matrix.T.reshape(-1))
        self.prog['proj_matrix'].value = tuple(proj_matrix.T.reshape(-1))

        # rendering
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()

        # read results
        raw = self.fbo.read(components=4, dtype='f4')
        buf = np.frombuffer(raw, dtype='float32').reshape((self.image_height, self.image_width, 4))
        normal = buf[:,:,:3]
        mask = (buf[:,:,3]!=1).astype(np.float32)

        data = self.fbo.read(attachment=-1, dtype='f4')
        buf = np.frombuffer(data, dtype='f4').reshape((self.image_height, self.image_width))
        depth = 2 * self.z_far * self.z_near / ((self.z_far + self.z_near) - (2 * buf - 1) * (self.z_far - self.z_near))
        depth[buf==1] = 0

        return depth, normal, mask


if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', help='ply or obj file to load', type=str)
    parser.add_argument('image_width', help='image width', type=int)
    parser.add_argument('image_height', help='image height', type=int)
    parser.add_argument('-i', '--intrinsics-file', required=True, help='intrinsics file to load', type=str)
    parser.add_argument('-e', '--extrinsics-file', required=True, help='extrinsics file to load', type=str)
    parser.add_argument('-d', '--depth-file', help='depth file (.npy) to save', type=str)
    parser.add_argument('-n', '--normal-file', help='normal file (.npy) to save', type=str)
    parser.add_argument('-m', '--mask-file', help='mask file to save', type=str)
    parser.add_argument('-s', '--shading-file', help='shading file to save', type=str)
    parser.add_argument('--up-sampling', help='upsampling ratio', type=float, default=1.0)
    args = parser.parse_args()
    
    mesh_file = args.mesh_file
    image_width, image_height = (args.image_width, args.image_height)

    # load camera parameters
    intrinsics = np.load(args.intrinsics_file)
    extrinsics = np.load(args.extrinsics_file)

    if args.up_sampling != 1.0:
        image_width = int(image_width * args.up_sampling)
        image_height = int(image_height * args.up_sampling)
        intrinsics[:2,:] *= args.up_sampling

    renderer = MeshGeometryRenderer(mesh_file, image_width, image_height, intrinsics=intrinsics)
    depth, normal, mask = renderer.render(extrinsics=extrinsics)

    # shading
    # assuminglambertian + co-located lighting
    rgb = np.tile(np.clip(normal[:,:,2:3], 0, 1), (1,1,3))


    if not (args.normal_file is None):
        assert args.normal_file.split('.')[-1] == 'npy'
        np.save(args.normal_file, normal)

    if not (args.depth_file is None):
        assert args.depth_file.split('.')[-1] == 'npy'
        np.save(args.depth_file, depth)

    if not (args.mask_file is None):
        if not args.mask_file.split('.')[-1] in ['exr', 'hdr', 'pfm']:
            mask = np.tile((mask.astype(np.uint8) * 255)[:,:,None], (1,1,3))
        cv2.imwrite(args.mask_file, mask)

    if not (args.shading_file is None):
        if not args.shading_file.split('.')[-1] in ['exr', 'hdr', 'pfm']:
            rgb = ((rgb**(1/2.2)) * 255).astype(np.uint8)
        cv2.imwrite(args.shading_file, rgb)