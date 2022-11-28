import numpy as np
import matplotlib.pyplot as plt
from .BRDFRead import read_brdf, normalize, theta_diff_index, theta_half_index_float, phi_diff_index_float
import cv2
import glob

def render_sphere(brdf, light, imsize=(512,512)):
    light = normalize(light)
    view = normalize(np.array([0.0,0.0,-1.0]))
    half = normalize(light + (1+1e-9)*view)
    theta_diff = np.arccos(np.clip(np.sum(half*view),-1,1))

    idx_theta_d = theta_diff_index(theta_diff)
    brdf_slice = brdf[:,idx_theta_d,:,:]

    #plt.imshow(np.clip(brdf_slice,0,1)**(1/2.2))
    #plt.ylabel('theta_h_index')
    #plt.xlabel('phi_d_index')
    #plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
    #plt.tick_params(bottom=False,left=False,right=False,top=False)
    #plt.box(False)
    #plt.show()

    u,v = np.meshgrid(range(imsize[0]), range(imsize[1]))
    x = 2.0 * (u + 0.5) / (imsize[0]) - 1.0
    y = 2.0 * (v + 0.5) / (imsize[0]) - 1.0
    r = np.sqrt(x**2+y**2)
    mask = (r <= 1.0).astype(np.float)
    z = -np.sqrt(np.clip(1-r**2,0,None))
    normal = np.array([x,y,z]).transpose((1,2,0))
    normal *= mask[:,:,None]

    theta_half = np.arccos(np.clip(np.sum(normal*half,axis=2),-1,1))
    b = normalize(np.cross(normal,half)+1e-9*np.array([1.0,0.0,0.0]), axis=2)
    a = normalize(np.cross(b,half), axis=2)
    cos_a = np.sum(a*light,axis=2)
    cos_b = np.sum(b*light,axis=2)
    phi_diff = np.arctan2(cos_b,cos_a)
    phi_diff[phi_diff<0.0] += np.pi

    cos_theta_in = np.clip(np.sum(normal*light,axis=2), -1, 1)

    grid_v = theta_half_index_float(theta_half).astype(np.float32)
    grid_u = phi_diff_index_float(phi_diff).astype(np.float32)
    src = np.concatenate([brdf_slice,brdf_slice,brdf_slice], axis=1).astype(np.float32)
    grid_u += brdf_slice.shape[1]
    img = cv2.remap(src, grid_u, grid_v,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE) * (np.clip(cos_theta_in,0,1) * mask)[:,:,None]

    img = np.clip(img,0,None)

    return img, mask

def render_cascade(brdf, increment = 30, imsize=(512,512)):
    width, height = imsize
    angles_deg = range(0, 180, increment)
    img = np.zeros((height, (width * len(angles_deg)) // 2 + width // 2, 3), dtype=np.float32)
    mask = np.zeros((height, (width * len(angles_deg)) // 2 + width // 2), dtype=np.float32)
    for i, angle_deg in enumerate(angles_deg):
        #print('angle(deg):', angle_deg)
        angle = np.radians(angle_deg)
        light = normalize(np.array([-np.sin(angle),0.0,-np.cos(angle)]))
        img_, mask_ = render_sphere(brdf, light, imsize=(width,height))
        offset = (i * width) // 2
        m = (mask_ > 0.0)
        img[:, offset:offset + width][m] = img_[m]
        mask[:, offset:offset + width][m] = mask_[m]
    return img, mask

def visualize_merl_as_sheres(brdf_path, increment=30):
    brdf = read_brdf(brdf_path)
    img, mask = render_cascade(brdf, increment = increment, imsize=(512,512))
    return img, mask


if __name__ == '__main__':
    if True:
        out_dir = 'cascade_spheres_merl'
        list_brdf_path = sorted(glob.glob('/home/kyamashita/data/BRDF/MERL/brdfs/*.binary'))
    else:
        out_dir = 'cascade_spheres_rgl'    
        list_brdf_path = sorted(glob.glob('/home/kyamashita/data/BRDF/rgl-merl/*.binary'))
    list_brdf_name = [l.split('/')[-1].split('.')[0] for l in list_brdf_path]

    for brdf_path, brdf_name in zip(list_brdf_path, list_brdf_name):
        brdf = read_brdf(brdf_path)

        img, mask = render_cascade(brdf, increment = 30, imsize=(512,512))
        cv2.imwrite(out_dir+'/'+brdf_name+'.exr', img[:,:,::-1])
        img = np.clip(img**(1/2.2), 0.0, 1.0)

        img = img / np.clip(np.max(img), 1e-9,1)
        img = np.concatenate([img,mask[:,:,None]], axis=-1)

        if False:
            plt.imshow(img)
            plt.title(brdf_name)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.show()

        img[:,:,:3] = img[:,:,:3][:,:,::-1]

        cv2.imwrite(out_dir+'/'+brdf_name+'.png', (255*img).astype(np.uint8))