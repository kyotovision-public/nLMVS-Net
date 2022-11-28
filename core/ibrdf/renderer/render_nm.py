import numpy as np

import torch
import torch.nn as nn

import numba
from numba import cuda
import math

import ctypes

from .analytical_brdfs import CosThetaIn, Phong, generateMERL

kBRDFSamplingResThetaH = 90
kBRDFSamplingResThetaD = 90
kBRDFSamplingResPhiD = 360

kRedScale = 1.0 / 1500.0
kGreenScale = 1.15 / 1500.0
kBlueScale = 1.66 / 1500.0

def CosThetaDiff(x, params):
    cos_theta_d = torch.cos(x[:,1:2])
    return cos_theta_d.repeat(1,3) # BS*3
merl_weight = torch.load('/'.join(__file__.split('/')[:-1])+'/merl_appearance_ratio.pt').float()
merl_weight *= generateMERL(CosThetaDiff, torch.tensor([0.0]))[0].float()

params = torch.tensor([1.0,0.0,0.0,0.0,0.0,0.0,0.0]).float()
lambert_basis = generateMERL(Phong, params)[0] * merl_weight
lambert_basis = lambert_basis / torch.sqrt(torch.sum(lambert_basis**2))

phong_alphas = torch.from_numpy((10.0*2**np.arange(10)).astype(np.float32))
phong_basis = []
for idx_alpha, alpha in enumerate(phong_alphas):
    params = torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,alpha.item()]).float()
    basis = generateMERL(Phong, params)[0] * merl_weight
    basis -= torch.sum(basis * lambert_basis) * lambert_basis
    basis = basis / torch.sqrt(torch.sum(basis**2))
    phong_basis.append(basis)
phong_basis = torch.stack(phong_basis, dim=0).float()

def matmul(m1, m2, dim1, dim2):
  if (dim1 is None) or (dim2 is None):
    return torch.matmul(m1, m2)
  d = m1.ndim
  m1_ = torch.transpose(torch.transpose(m1, dim1, d-2), dim2, d-1)
  m2_ = torch.transpose(torch.transpose(m2, dim1, d-2), dim2, d-1)
  return torch.transpose(torch.transpose(torch.matmul(m1_, m2_), dim1, d-2), dim2, d-1)

# device functions
@cuda.jit(device=True)
def clip_device(x, min_, max_):
    return min(max(min_, x), max_)

@cuda.jit(device=True)
def van_der_corput_device(n,b):
    g = 0
    c = 1.0
    while (n > 0) and (c > 1e-16):
        c /= b
        g += (n % b) * c
        n = n // b
    return g

@cuda.jit(device=True)
def sample_dimension_halton(index, dim):
    prime_numbers = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)#, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
    permutation_table = (
        1, 0, 1, 0, 2, 4, 0, 1, 2, 3, 2, 5, 0, 6, 3, 1, 4, 6, 10, 2, 3, 8, 4, 5, 7, 1, 0, 9, 5, 9, 
        3, 7, 8, 1, 4, 12, 6, 11, 10, 0, 2, 3, 2, 11, 10, 5, 16, 8, 12, 7, 4, 13, 0, 9, 14, 6, 1, 
        15, 11, 7, 0, 15, 18, 13, 9, 1, 5, 12, 6, 16, 17, 10, 4, 3, 8, 2, 14, 18, 14, 16, 4, 0, 17, 
        5, 15, 2, 1, 21, 3, 22, 10, 7, 11, 12, 13, 20, 9, 8, 6, 19, 16, 20, 18, 22, 26, 3, 10, 24, 
        13, 6, 25, 15, 23, 7, 14, 4, 5, 11, 28, 1, 27, 17, 9, 0, 8, 2, 12, 19, 21
    )
    permutation_offset = (
        0, 2, 5, 10, 17, 28, 41, 58, 77, 100#, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568
    )

    n = index
    p = prime_numbers[dim]
    ofs = permutation_offset[dim]
    g = 0
    c = 1.0
    while (n > 0) and (c > 1e-16):
        c /= p
        digit = n % p
        digit = permutation_table[ofs+digit]
        g += digit * c
        n = n // p
    return g
    #return van_der_corput_device(index, prime_numbers[dim])

@cuda.jit(device=True)
def dot_device(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] +v1[2] * v2[2]

@cuda.jit(device=True)
def reflect_device(v, n):
    vDn = dot_device(v,n)
    x = 2*vDn*n[0] - v[0]
    y = 2*vDn*n[1] - v[1]
    z = 2*vDn*n[2] - v[2]
    return x,y,z

@cuda.jit(device=True)
def binsearch_device(arr, val):
    if val < arr[0]:
        return 0
    idx1 = 0
    idx2 = len(arr) - 1
    while idx2 > (idx1 + 1):
        idx_m = (idx1+idx2) // 2
        if arr[idx_m] > val:
            idx2 = idx_m
        else:
            idx1 = idx_m
    return idx2

@cuda.jit(device=True)
def vec2polar_device(x,y,z):
    theta = math.acos(min(max(-1,z),1))
    phi = math.atan2(y, x)
    if phi < 0:
      phi += 2.0*np.pi
    return phi, theta


@cuda.jit(device=True)
def phong_sample_hemisphere_device(z1,z2,alpha):
    cosTheta = z1**(1.0/(alpha+1))
    sinTheta = math.sqrt(max(0.0, 1.0-cosTheta**2))
    pdf = (alpha+1)/(2*np.pi)*(z1**((alpha)/(alpha+1)))
    phi = 2.0 * np.pi * z2
    x = sinTheta * math.cos(phi)
    y = sinTheta * math.sin(phi)
    z = cosTheta
    return x,y,z,pdf

@cuda.jit(device=True)
def rotate_polar_device(x,y,z,phi,theta):
    x1 = math.cos(theta) * x + math.sin(theta) * z
    y1 = y
    z1 = -math.sin(theta) * x + math.cos(theta) * z

    x2 = math.cos(phi) * x1 - math.sin(phi) * y1
    y2 = math.sin(phi) * x1 + math.cos(phi) * y1
    z2 = z1
    return x2,y2,z2

@cuda.jit(device=True)
def phong_pdf_device(cosTheta,alpha):
    pdf = (alpha+1)/(2*math.pi) * cosTheta**alpha
    return pdf

@cuda.jit(device=True)
def half_vector_device(v1, v2):
    x = v1[0] + v2[0]
    y = v1[1] + v2[1]
    z = v1[2] + v2[2]
    l = math.sqrt(x*x+y*y+z*z)+1e-12
    return x/l, y/l, z/l

@cuda.jit(device=True)
def cross_device(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2
    x = y1*z2 - y2*z1
    y = z1*x2 - z2*x1
    z = x1*y2 - x2*y1
    return x,y,z

@cuda.jit(device=True)
def cross_grad_v1_device(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2
    return ((0, -z2, y2), (z2,0,-x2),(-y2,x2,0))

@cuda.jit(device=True)
def scale_vector_device(scale, v):
    return scale * v[0], scale * v[1], scale * v[2]

@cuda.jit(device=True)
def add_vector_device(v1, v2):
    return v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]

@cuda.jit(device=True)
def normalize_device(v):
    x,y,z = v
    l = math.sqrt(x*x+y*y+z*z)+1e-12
    return x/l, y/l, z/l

@cuda.jit(device=True)
def normalize_grad_device(v):
    x,y,z = v
    l = math.sqrt(x*x+y*y+z*z)+1e-12
    c1 = 1 / l
    c2 = 1 / (l**3)
    r1 = (c1-c2*x*x, -c2*x*y, -c2*x*z)
    r2 = (-c2*x*y, c1-c2*y*y, -c2*y*z)
    r3 = (-c2*x*z, -c2*y*z, c1-c2*z*z)

    return (r1,r2,r3)

@cuda.jit(device=True)
def matmul333_device(a,b):
    c00 = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0]
    c01 = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1]
    c02 = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]
    c10 = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0]
    c11 = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1]
    c12 = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]    
    c20 = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0]
    c21 = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1]
    c22 = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]
    return ((c00,c01,c02), (c10,c11,c12), (c20,c21,c22))

@cuda.jit(device=True)
def mul_vec_mat_device(v,mat):
    c00 = v[0] * mat[0][0] + v[1] * mat[1][0] + v[2] * mat[2][0]
    c01 = v[0] * mat[0][1] + v[1] * mat[1][1] + v[2] * mat[2][1]
    c02 = v[0] * mat[0][2] + v[1] * mat[1][2] + v[2] * mat[2][2]
    return (c00,c01,c02)

@cuda.jit(device=True)
def vec2halfdiff_device(v_in, v_out, n):
    h = half_vector_device(v_in, v_out)
    theta_h = math.acos(min(max(-1,dot_device(h,n)),1))
    theta_d = math.acos(min(max(-1,dot_device(h,v_in)),1))
    a = add_vector_device(scale_vector_device(-1,n), scale_vector_device(dot_device(n,h), h))
    if dot_device(a,a) <= 0.001**2:
        phi_d = 0.0
    else:
        a = normalize_device(a)
        b = cross_device(h,a)
        b = normalize_device(b)
        x = dot_device(a,v_in)
        y = dot_device(b,v_in)
        phi_d = math.atan2(y,x)
    #if theta_h == 0.0:
    #    return theta_h, theta_d, 0.0
    #b = cross_device(n,h)
    #b = normalize_device(b)
    #a = cross_device(b,h)
    #b = cross_device(h,a)
    #x = dot_device(a,v_in)
    #y = dot_device(b,v_in)
    #phi_d = math.atan2(y,x)
    return theta_h, theta_d, phi_d

# ret 3*3 matrix
# TODO: grad through phi_d is incorrect!
@cuda.jit(device=True)
def grad_n_vec2halfdiff_device(v_in, v_out, n):
    h = half_vector_device(v_in, v_out)
    s = math.sqrt(max(1e-20, 1.0-dot_device(h,n)**2))
    grad_theta_h = scale_vector_device(-1.0/s, h)
    theta_h = math.acos(min(max(-1,dot_device(h,n)),1))
    grad_theta_d = (0.0,0.0,0.0)
    if theta_h == 0.0:
        grad_phi_d = (0.0,0.0,0.0)
        return (grad_theta_h, grad_theta_d, grad_phi_d)
    
    b0 = cross_device(n,h)
    grad_b0 = cross_grad_v1_device(n,h)
    grad_b0 = matmul333_device(normalize_grad_device(b0), grad_b0)
    b0 = normalize_device(b0)

    a = cross_device(b0,h)
    grad_a = matmul333_device(cross_grad_v1_device(b0,h), grad_b0)
    
    b = cross_device(h,a)
    grad_b = matmul333_device(cross_grad_v1_device(a,scale_vector_device(-1,h)), grad_a)

    x = dot_device(a,v_in)
    grad_x = mul_vec_mat_device(v_in, grad_a)
    y = dot_device(b,v_in)
    grad_y = mul_vec_mat_device(v_in, grad_b)

    phi_d = math.atan2(y,x)
    s = max(1e-20, x*x+y*y)
    grad_phi_d_x = scale_vector_device(-y/s, grad_x)
    grad_phi_d_y = scale_vector_device(x/s, grad_y)
    grad_phi_d = add_vector_device(grad_phi_d_x, grad_phi_d_y)
    return (grad_theta_h, grad_theta_d, grad_phi_d) # 3x3 matrix

@cuda.jit(device=True)
def halfdiff2index_device(theta_h, theta_d, phi_d):
    if theta_h <= 0.0:
        idx1 = 0
    else:
        thetaHalfDeg = ((theta_h / (np.pi / 2.0)) * kBRDFSamplingResThetaH)
        temp = thetaHalfDeg * kBRDFSamplingResThetaH
        idx1 = min(max(0, math.sqrt(temp)), kBRDFSamplingResThetaH)
    
    tmp = theta_d / (np.pi * 0.5) * kBRDFSamplingResThetaD
    idx2 = min(max(0, tmp), kBRDFSamplingResThetaD)

    if phi_d < 0.0:
        phi_d += np.pi
    tmp = phi_d / np.pi * kBRDFSamplingResPhiD / 2
    idx3 = min(max(0, tmp), kBRDFSamplingResPhiD/2)

    return idx1, idx2, idx3

@cuda.jit(device=True)
def grad_halfdiff2index_device(theta_h, theta_d, phi_d):
    if theta_h <= 0.0:
        grad_idx1 = 0.0
    else:
        thetaHalfDeg = ((theta_h / (np.pi / 2.0)) * kBRDFSamplingResThetaH)
        temp = thetaHalfDeg * kBRDFSamplingResThetaH
        idx1 = min(max(0, math.sqrt(temp)), kBRDFSamplingResThetaH)
        if idx1 == kBRDFSamplingResThetaH:
            grad_idx1 = 0.0
        else:
            grad_idx1 = 0.5* kBRDFSamplingResThetaH**2/(np.pi / 2.0) / (idx1+1e-3)
    
    tmp = theta_d / (np.pi * 0.5) * kBRDFSamplingResThetaD
    if (tmp < 0.0) or (tmp >= kBRDFSamplingResThetaD):
        grad_idx2 = 0.0
    else:
        grad_idx2 = kBRDFSamplingResThetaD / (np.pi * 0.5)

    if phi_d < 0.0:
        phi_d += np.pi
    tmp = phi_d / np.pi * kBRDFSamplingResPhiD / 2
    if (tmp < 0.0) or (tmp >= kBRDFSamplingResPhiD/2):
        grad_idx3 = 0.0
    else:
        grad_idx3 = kBRDFSamplingResPhiD / 2 / np.pi

    return (grad_idx1, grad_idx2, grad_idx3) # 3D vector

# brdf : (kBRDFSamplingResThetaH, kBRDFSamplingResThetaD, kBRDFSamplingResThetaH/2)
@cuda.jit(device=True)
def lookup_brdf_val_device(brdf, idx1, idx2, idx3):
    idx1 = min(max(0, idx1-0.5), kBRDFSamplingResThetaH-1.001)
    idx2 = min(max(0, idx2-0.5), kBRDFSamplingResThetaD-1.001)
    idx3 = min(max(0, idx3-0.5), kBRDFSamplingResPhiD/2-1.001)
    
    dif1 = min(max(0.0, idx1-int(idx1)), 1.0)
    dif2 = min(max(0.0, idx2-int(idx2)), 1.0)
    dif3 = min(max(0.0, idx3-int(idx3)), 1.0)

    # interp along dim 2
    a = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)]
    b = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)]
    c = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)+1]
    d = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)+1]

    # interp along dim 1
    val = (1.0-dif3)*(1.0-dif1) * a + (1.0-dif3)*dif1 * b + dif3*(1.0-dif1) * c + dif3*dif1 * d

    # interp along dim 3
    return max(0, val)

# brdf : (kBRDFSamplingResThetaH, kBRDFSamplingResThetaD, kBRDFSamplingResThetaH/2)
@cuda.jit(device=True)
def lookup_brdf_val_with_grad_idx_device(brdf, idx1, idx2, idx3):
    idx1 = min(max(0, idx1-0.5), kBRDFSamplingResThetaH-1.001)
    idx2 = min(max(0, idx2-0.5), kBRDFSamplingResThetaD-1.001)
    idx3 = min(max(0, idx3-0.5), kBRDFSamplingResPhiD/2-1.001)
    
    dif1 = min(max(0.0, idx1-int(idx1)), 1.0)
    dif2 = min(max(0.0, idx2-int(idx2)), 1.0)
    dif3 = min(max(0.0, idx3-int(idx3)), 1.0)

    # interp along dim 2
    a = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)]
    b = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)]
    c = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)+1]
    d = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)+1]
    grad_idx2 = 0.0 # ignore as it doesn't affect to grad_normal

    # interp along dim 1
    val = (1.0-dif3)*(1.0-dif1) * a + (1.0-dif3)*dif1 * b + dif3*(1.0-dif1) * c + dif3*dif1 * d
    if val < 0.0:
        grad_idx1 = grad_idx3 = 0.0
    else:
        grad_idx1 = -(1.0-dif3) * a + (1.0-dif3) * b - dif3 * c + dif3 * d
        grad_idx3 = -(1.0-dif1) * a - dif1 * b + (1.0-dif1) * c + dif1 * d

    # interp along dim 3
    return max(0, val), (grad_idx1, grad_idx2, grad_idx3) # scalar, 3D vector


# brdf : (kBRDFSamplingResThetaH, kBRDFSamplingResThetaD, kBRDFSamplingResThetaH/2)
@cuda.jit(device=True)
def lookup_brdf_val_with_grad_brdf_device(brdf, idx1, idx2, idx3):
    idx1 = min(max(0, idx1-0.5), kBRDFSamplingResThetaH-1.001)
    idx2 = min(max(0, idx2-0.5), kBRDFSamplingResThetaD-1.001)
    idx3 = min(max(0, idx3-0.5), kBRDFSamplingResPhiD/2-1.001)
    
    dif1 = min(max(0.0, idx1-int(idx1)), 1.0)
    dif2 = min(max(0.0, idx2-int(idx2)), 1.0)
    dif3 = min(max(0.0, idx3-int(idx3)), 1.0)

    # interp along dim 2
    a = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)]
    b = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)]
    c = (1.0-dif2) * brdf[int(idx1),int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1),int(idx2)+1,int(idx3)+1]
    d = (1.0-dif2) * brdf[int(idx1)+1,int(idx2),int(idx3)+1] + dif2 * brdf[int(idx1)+1,int(idx2)+1,int(idx3)+1]
    
    # interp along dim 1
    val = (1.0-dif3)*(1.0-dif1) * a + (1.0-dif3)*dif1 * b + dif3*(1.0-dif1) * c + dif3*dif1 * d
    if val < 0.0:
        grad_brdf = (
            (
                (0.0, 0.0), # [0,0,0], [0,0,1]
                (0.0, 0.0)  # [0,1,0], [0,1,1]
            ),
            (
                (0.0, 0.0), # [1,0,0], [1,0,1]
                (0.0, 0.0)  # [1,1,0], [1,1,1]
            ),            
        )
    else:
        grad_brdf = (
            (
                ((1.0-dif2)*(1.0-dif3)*(1.0-dif1), (1.0-dif2)*dif3*(1.0-dif1)), # [0,0,0], [0,0,1]
                (dif2*(1.0-dif3)*(1.0-dif1), dif2*dif3*(1.0-dif1))  # [0,1,0], [0,1,1]
            ),
            (
                ((1.0-dif2)*(1.0-dif3)*dif1, (1.0-dif2)*dif3*dif1), # [1,0,0], [1,0,1]
                (dif2*(1.0-dif3)*dif1, dif2*dif3*dif1)  # [1,1,0], [1,1,1]
            ),            
        )      

    # interp along dim 3
    return max(0, val), (int(idx1),int(idx2),int(idx3)), grad_brdf # scalar, 3D vector

@cuda.jit(device=True)
def brdf_eval_device(brdf, v_in, v_out, n):
    theta_h, theta_d, phi_d = vec2halfdiff_device(v_in, v_out, n)
    idx1, idx2, idx3 = halfdiff2index_device(theta_h, theta_d, phi_d)
    
    red = kRedScale * lookup_brdf_val_device(brdf[0], idx1, idx2, idx3)
    green = kGreenScale * lookup_brdf_val_device(brdf[1], idx1, idx2, idx3)
    blue = kBlueScale * lookup_brdf_val_device(brdf[2], idx1, idx2, idx3)

    return red, green, blue

@cuda.jit(device=True)
def brdf_eval_with_grad_n_device(brdf, v_in, v_out, n):
    theta_h, theta_d, phi_d = vec2halfdiff_device(v_in, v_out, n)
    grad_ang = grad_n_vec2halfdiff_device(v_in, v_out, n) # 3x3 matrix

    idx1, idx2, idx3 = halfdiff2index_device(theta_h, theta_d, phi_d)
    grad_idx = grad_halfdiff2index_device(theta_h, theta_d, phi_d) # 3D vector
    
    val, g = lookup_brdf_val_with_grad_idx_device(brdf[0], idx1, idx2, idx3)
    red = kRedScale * val
    grad_red =  mul_vec_mat_device((g[0]*grad_idx[0],g[1]*grad_idx[1],g[2]*grad_idx[2]), grad_ang)
    grad_red = scale_vector_device(kRedScale, grad_red)
    
    val, g = lookup_brdf_val_with_grad_idx_device(brdf[1], idx1, idx2, idx3)
    green = kGreenScale * val
    grad_green =  mul_vec_mat_device((g[0]*grad_idx[0],g[1]*grad_idx[1],g[2]*grad_idx[2]), grad_ang)
    grad_green = scale_vector_device(kGreenScale, grad_green)
    
    val, g = lookup_brdf_val_with_grad_idx_device(brdf[2], idx1, idx2, idx3)
    blue = kBlueScale * val
    grad_blue =  mul_vec_mat_device((g[0]*grad_idx[0],g[1]*grad_idx[1],g[2]*grad_idx[2]), grad_ang)
    grad_blue = scale_vector_device(kBlueScale, grad_blue)

    return (red, green, blue), (grad_red, grad_green, grad_blue)

@cuda.jit(device=True)
def scale_grad_brdf(s, grad):
    return (
        (
            (s*grad[0][0][0], s*grad[0][0][1]), # [0,0,0], [0,0,1]
            (s*grad[0][1][0], s*grad[0][1][1])  # [0,1,0], [0,1,1]
        ),
        (
            (s*grad[1][0][0], s*grad[1][0][1]), # [1,0,0], [1,0,1]
            (s*grad[1][1][0], s*grad[1][1][1])  # [1,1,0], [1,1,1]
        ),        
    )

@cuda.jit(device=True)
def brdf_eval_with_grad_brdf_device(brdf, v_in, v_out, n):
    theta_h, theta_d, phi_d = vec2halfdiff_device(v_in, v_out, n)
    idx1, idx2, idx3 = halfdiff2index_device(theta_h, theta_d, phi_d)
    
    val, indices, grad_red = lookup_brdf_val_with_grad_brdf_device(brdf[0], idx1, idx2, idx3)
    red = kRedScale * val
    grad_red = scale_grad_brdf(kRedScale, grad_red)
    
    val, indices, grad_green = lookup_brdf_val_with_grad_brdf_device(brdf[1], idx1, idx2, idx3)
    green = kGreenScale * val
    grad_green = scale_grad_brdf(kGreenScale, grad_green)
    
    val, indices, grad_blue = lookup_brdf_val_with_grad_brdf_device(brdf[2], idx1, idx2, idx3)
    blue = kBlueScale * val
    grad_blue = scale_grad_brdf(kBlueScale, grad_blue)

    return (red, green, blue), indices, (grad_red, grad_green, grad_blue)

@cuda.jit(device=True)
def sample_incident_direction(rand_values, n, v, pdf, cdf1, cdf2, alpha, use_sample_light):
    He,We = pdf.shape
    phi_n, theta_n = vec2polar_device(n[0], n[1], n[2])
    r = reflect_device(v, n)
    phi_r, theta_r = vec2polar_device(r[0], r[1], r[2])

    vDn = dot_device(v, n)
    if (vDn < 0.0) or ((n[0] == 0.0) and (n[1] == 0.0) and (n[2] == 0.0)):
        return (0.0, 0.0, 0.0), 0.0

    z0, z1, z2, z3, z4 = rand_values[0], rand_values[1], rand_values[2], rand_values[3], rand_values[4]

    if use_sample_light:
        z0 = 3 * z0
    else:
        z0 = 2 * z0    

    pdf_d = -1.0
    pdf_s = -1.0
    pdf_i = -1.0

    if z0 < 1.0:
        # sample diffuse brdf
        x,y,z,p = phong_sample_hemisphere_device(z1,z2,1.0)
        l = rotate_polar_device(x,y,z,phi_n,theta_n)
        pdf_d = p
    elif z0 < 2.0:
        # sample specular brdf
        x,y,z,p = phong_sample_hemisphere_device(z1,z2,alpha)
        l = rotate_polar_device(x,y,z,phi_r,theta_r)  
        pdf_s = p
    else:
        # sample light pdf
        idx_ve = binsearch_device(cdf1, z1)
        idx_ue = binsearch_device(cdf2[idx_ve], z2)

        phi = 2.0 * np.pi * (idx_ue + z3) / We
    
        theta0 = np.pi * idx_ve / He
        cosTheta = math.cos(theta0) + (math.cos(theta0+np.pi/He)-math.cos(theta0)) * z4
        sinTheta = math.sqrt(max(0,1.0-cosTheta**2))

        l = (sinTheta * math.cos(phi), -sinTheta * math.sin(phi), cosTheta)

        pdf_i = pdf[idx_ve,idx_ue]

    # evaluate pdf
    if pdf_d < 0.0:
        lDn = dot_device(l,n)
        pdf_d = phong_pdf_device(max(0,lDn),1.0)
    if pdf_s < 0.0:
        lDr = dot_device(l,r)
        pdf_s = phong_pdf_device(max(0,lDr),alpha)
    if pdf_i < 0.0:
        phi, theta = vec2polar_device(l[0], -l[1], l[2])
        idx_ve = int(clip_device(theta / np.pi * He, 0.0, He - 0.0001))
        idx_ue = int(clip_device(phi * We / (2.0*np.pi), 0.0, We - 0.0001))
        pdf_i = pdf[idx_ve,idx_ue]

    if use_sample_light:
        p = (pdf_d + pdf_s + pdf_i) / 3
    else:
        p = (pdf_d + pdf_s) / 2
    return l, p

# normal : BS * 3 * H * W
# view   : BS * 3 * H * W
# brdf   : BS * 3 * D * D * D
# envmap : BS * 3 * He * We
# wi     : BS * H * W * num_sample * 3
# pdf    : BS * H * W * num_sample
# result : BS * 3 * H * W
# grad_normal : BS * 3 * 3 * H * W
# BS * H * W blocks, min(num_sample,512) threads
@cuda.jit
def render_forward_kernel(
    # inputs
    normal, view, brdf, envmap, 
    # sampling parameters
    envmap_pdf, envmap_cdf1, envmap_cdf2, specular_alpha, num_sample, seed,
    # result buffer
    result, grad_normal
):
    BS = normal.shape[0]
    H = normal.shape[2]
    W = normal.shape[3]
    He = envmap.shape[2]
    We = envmap.shape[3]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    n = normal[idx_batch,:,idx_v,idx_u]
    view   = view[idx_batch,:,idx_v,idx_u]

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    envmap_pdf = envmap_pdf[idx_batch]
    envmap_cdf1 = envmap_cdf1[idx_batch]
    envmap_cdf2 = envmap_cdf2[idx_batch]
    specular_alpha = specular_alpha[idx_batch]

    brdf = brdf[idx_batch]

    result_tmp = cuda.local.array(shape=(3), dtype=numba.float32)
    grad_normal_tmp = cuda.local.array(shape=(3,3), dtype=numba.float32)
    for i in range(3):
        result_tmp[i] = 0.0
        for j in range(3):
            grad_normal_tmp[i,j] = 0.0

    vDn = dot_device(view, n)
    if (vDn > 0.0) and (not ((n[0] == 0.0) and (n[1] == 0.0) and (n[2] == 0.0))):
        idx = idx_thread
        while idx < num_sample:
            # sample incident direction
            idx_sample = seed + ((idx_batch * H + idx_v) * W + idx_u) * num_sample + idx
            z0 = sample_dimension_halton(idx_sample, 0)
            z1 = sample_dimension_halton(idx_sample, 1)
            z2 = sample_dimension_halton(idx_sample, 2)
            z3 = sample_dimension_halton(idx_sample, 3)
            z4 = sample_dimension_halton(idx_sample, 4)
            rand_values = (z0, z1, z2, z3, z4)
            l, p = sample_incident_direction(
                rand_values, n, view, 
                envmap_pdf, envmap_cdf1, envmap_cdf2, specular_alpha, True
            )
            phi_i, theta_i = vec2polar_device(l[0],-l[1],l[2])
            
            ue = clip_device(phi_i / (2.0*np.pi) * We, 0.0, We - 0.0001)
            ve = clip_device(theta_i / np.pi * He, 0.0, He - 0.0001)
            Li = envmap[idx_batch,:,int(ve),int(ue)]

            lDn = dot_device(l, n)
            if (lDn > 0.0) and ((Li[0] > 0.0) or (Li[1] > 0.0) or (Li[2] > 0.0)):
                # sample brdf
                brdf_val, grad_brdf_n = brdf_eval_with_grad_n_device(brdf, l, view, n)              
            
                for i in range(3):
                    result_tmp[i] += Li[i] * brdf_val[i] * lDn / (p+1e-15) / num_sample
                    for j in range(3):
                        grad_normal_tmp[i,j] += Li[i] * brdf_val[i] * l[j] / (p+1e-15) / num_sample
                        grad_normal_tmp[i,j] += Li[i] * grad_brdf_n[i][j] * lDn / (p+1e-15) / num_sample
      
            idx += blockdim

        # write to result buffer
        for i in range(3):
            cuda.atomic.add(result, (idx_batch,i,idx_v,idx_u), result_tmp[i])
            for j in range(3):
                cuda.atomic.add(grad_normal, (idx_batch,i,j,idx_v,idx_u), grad_normal_tmp[i,j])

# normal : BS * 3 * H * W
# view   : BS * 3 * H * W
# brdf   : BS * 3 * D * D * D
# envmap : BS * 3 * He * We
# wi     : BS * H * W * num_sample * 3
# pdf    : BS * H * W * num_sample
# result : BS * 3 * H * W
# grad_normal : BS * 3 * 3 * H * W
# BS * H * W blocks, min(num_sample,512) threads
@cuda.jit
def render_backward_kernel(
    # inputs
    normal, view, brdf, envmap,  
    # sampling parameters
    envmap_pdf, envmap_cdf1, envmap_cdf2, specular_alpha, num_sample, seed,
    # input gradient and output buffer
    grad_output, grad_brdf, grad_envmap, 
    requires_grad_brdf, requires_grad_envmap
):
    BS = normal.shape[0]
    H = normal.shape[2]
    W = normal.shape[3]
    He = envmap.shape[2]
    We = envmap.shape[3]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    n = normal[idx_batch,:,idx_v,idx_u]
    view   = view[idx_batch,:,idx_v,idx_u]

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    envmap_pdf = envmap_pdf[idx_batch]
    envmap_cdf1 = envmap_cdf1[idx_batch]
    envmap_cdf2 = envmap_cdf2[idx_batch]
    specular_alpha = specular_alpha[idx_batch]

    brdf = brdf[idx_batch]

    grad_output = grad_output[idx_batch,:,idx_v,idx_u]
    
    vDn = dot_device(view, n)
    if (vDn > 0.0) and (not ((n[0] == 0.0) and (n[1] == 0.0) and (n[2] == 0.0))):
        idx = idx_thread
        while idx < num_sample:
            # sample incident direction
            idx_sample = seed + ((idx_batch * H + idx_v) * W + idx_u) * num_sample + idx
            z0 = sample_dimension_halton(idx_sample, 0)
            z1 = sample_dimension_halton(idx_sample, 1)
            z2 = sample_dimension_halton(idx_sample, 2)
            z3 = sample_dimension_halton(idx_sample, 3)
            z4 = sample_dimension_halton(idx_sample, 4)
            rand_values = (z0, z1, z2, z3, z4)
            l, p = sample_incident_direction(
                rand_values, n, view, 
                envmap_pdf, envmap_cdf1, envmap_cdf2, specular_alpha, True
            )
            phi_i, theta_i = vec2polar_device(l[0],-l[1],l[2])

            ue = clip_device(phi_i / (2.0*np.pi) * We, 0.0, We - 0.0001)
            ve = clip_device(theta_i / np.pi * He, 0.0, He - 0.0001)
            Li = envmap[idx_batch,:,int(ve),int(ue)]
            
            lDn = dot_device(l, n)
            if (lDn > 0.0) and ((Li[0] > 0.0) or (Li[1] > 0.0) or (Li[2] > 0.0)):
                # sample brdf
                #brdf_val = brdf_eval_device(brdf, wi[idx], view, n)
                brdf_val, indices, grad_brdf_brdf = brdf_eval_with_grad_brdf_device(brdf, l, view, n)              
                
                for i in range(3):
                    #result_tmp[idx_thread][i] += Li[i] * brdf_val[i] * lDn / (pdf[idx]+1e-15) / num_sample
                    if requires_grad_envmap:
                        grad_e = grad_output[i] * brdf_val[i] * lDn / (p+1e-15) / num_sample
                        cuda.atomic.add(grad_envmap, (idx_batch,i,int(ve),int(ue)), grad_e)
                    if requires_grad_brdf:
                        for j in range(2):
                            for k in range(2):
                                for l in  range(2):
                                    grad_b = grad_output[i] * grad_brdf_brdf[i][j][k][l] *  Li[i] * lDn / (p+1e-15) / num_sample
                                    cuda.atomic.add(grad_brdf, (idx_batch,i,indices[0]+j,indices[1]+k,indices[2]+l), grad_b)
      
            idx += blockdim

# normal : BS * 3 * H * W
# view   : BS * 3 * H * W
# brdf   : BS * 3 * D * D * D
# envmap : BS * 3 * He * We
# result : BS * 3 * H * W
# grad_normal : BS * 3 * 3 * H * W
# BS * H * W blocks, min(He*We,512) threads
@cuda.jit
def render_forward_bf_kernel(normal, view, brdf, envmap, result, grad_normal):
    BS = normal.shape[0]
    H = normal.shape[2]
    W = normal.shape[3]
    He = envmap.shape[2]
    We = envmap.shape[3]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    n = normal[idx_batch,:,idx_v,idx_u]
    view   = view[idx_batch,:,idx_v,idx_u]

    brdf = brdf[idx_batch]

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    #idx_rand = idx_thread + ((idx_batch * H + idx_v) * W + idx_u) * blockdim

    wi = cuda.local.array(shape=(3), dtype=numba.float32)
    
    result_tmp = cuda.local.array(shape=(3), dtype=numba.float32)
    grad_normal_tmp = cuda.local.array(shape=(3,3), dtype=numba.float32)
    for i in range(3):
        result_tmp[i] = 0.0
        for j in range(3):
            grad_normal_tmp[i,j] = 0.0

    vDn = dot_device(view, n)
    if (vDn > 0.0) and (not ((n[0] == 0.0) and (n[1] == 0.0) and (n[2] == 0.0))):
        idx_ve = idx_thread
        while idx_ve < He:
            theta_i = (idx_ve + 0.5) / He * np.pi
            cosTheta_i = math.cos(theta_i)
            sinTheta_i = math.sin(theta_i)

            dry = math.cos(theta_i - 0.5/He*np.pi) - math.cos(theta_i + 0.5/He*np.pi) 
            drx = 2.0*np.pi/We
            dwi = drx * dry

            for idx_ue in range(We):
                Li = envmap[idx_batch,:,idx_ve,idx_ue]

                phi_i = (We - (idx_ue + 0.5)) / We * 2.0 * np.pi
                wi[0] = sinTheta_i * math.cos(phi_i)
                wi[1] = sinTheta_i * math.sin(phi_i)
                wi[2] = cosTheta_i

                lDn = dot_device(wi, n)
                if (lDn > 0.0) and ((Li[0] > 0.0) or (Li[1] > 0.0) or (Li[2] > 0.0)):
                    # sample brdf
                    #brdf_val = brdf_eval_device(brdf, wi, view, n)
                    brdf_val, grad_brdf_n = brdf_eval_with_grad_n_device(brdf, wi, view, n)

                    for i in range(3):
                        result_tmp[i] += Li[i] * brdf_val[i] * lDn * dwi                    
                        for j in range(3):
                            grad_normal_tmp[i,j] += Li[i] * brdf_val[i] * wi[j] * dwi
                            grad_normal_tmp[i,j] += Li[i] * grad_brdf_n[i][j] * lDn *dwi

            idx_ve += blockdim

        # write to result buffer
        for i in range(3):
            cuda.atomic.add(result, (idx_batch,i,idx_v,idx_u), result_tmp[i])
            for j in range(3):
                cuda.atomic.add(grad_normal, (idx_batch,i,j,idx_v,idx_u), grad_normal_tmp[i,j])

@cuda.jit
def render_backward_bf_kernel(normal, view, brdf, envmap, grad_output, grad_brdf, grad_envmap, requires_grad_brdf, requires_grad_envmap):
    BS = normal.shape[0]
    H = normal.shape[2]
    W = normal.shape[3]
    He = envmap.shape[2]
    We = envmap.shape[3]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    n = normal[idx_batch,:,idx_v,idx_u]
    view   = view[idx_batch,:,idx_v,idx_u]

    brdf = brdf[idx_batch]

    grad_output = grad_output[idx_batch,:,idx_v,idx_u]

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    #idx_rand = idx_thread + ((idx_batch * H + idx_v) * W + idx_u) * blockdim

    wi = cuda.local.array(shape=(3), dtype=numba.float32)

    vDn = dot_device(view, n)
    grad_output_is_nonzero = (grad_output[0] != 0) or (grad_output[1] != 0) or (grad_output[2] != 0)
    if grad_output_is_nonzero and (vDn > 0.0) and (not ((n[0] == 0.0) and (n[1] == 0.0) and (n[2] == 0.0))):
        idx_ve = idx_thread
        while idx_ve < He:
            theta_i = (idx_ve + 0.5) / He * np.pi
            cosTheta_i = math.cos(theta_i)
            sinTheta_i = math.sin(theta_i)

            dry = math.cos(theta_i - 0.5/He*np.pi) - math.cos(theta_i + 0.5/He*np.pi) 
            drx = 2.0*np.pi/We
            dwi = drx * dry

            for idx_ue in range(We):
                Li = envmap[idx_batch,:,idx_ve,idx_ue]

                phi_i = (We - (idx_ue + 0.5)) / We * 2.0 * np.pi
                wi[0] = sinTheta_i * math.cos(phi_i)
                wi[1] = sinTheta_i * math.sin(phi_i)
                wi[2] = cosTheta_i

                lDn = dot_device(wi, n)
                if (lDn > 0.0) and ((Li[0] > 0.0) or (Li[1] > 0.0) or (Li[2] > 0.0)):
                    # sample brdf
                    #brdf_val = brdf_eval_device(brdf, wi, view, n)
                    brdf_val, indices, grad_brdf_brdf = brdf_eval_with_grad_brdf_device(brdf, wi, view, n)              

                    for i in range(3):
                        #result_tmp[idx_thread][i] += Li[i] * brdf_val[i] * lDn * dwi
                        if requires_grad_envmap:
                            grad_e = grad_output[i] * brdf_val[i] * lDn * dwi
                            cuda.atomic.add(grad_envmap, (idx_batch,i,idx_ve,idx_ue), grad_e)
                        if requires_grad_brdf:
                            for j in range(2):
                                for k in range(2):
                                    for l in  range(2):
                                        grad_b = grad_output[i] * grad_brdf_brdf[i][j][k][l] *  Li[i] * lDn * dwi
                                        cuda.atomic.add(grad_brdf, (idx_batch,i,indices[0]+j,indices[1]+k,indices[2]+l), grad_b)
      
            idx_ve += blockdim

# code from https://gist.github.com/t-vi/2f4fe23a5b473b9dceb95b163378b4d5
# Since PyTorch doesn't support CUDA Array Interface (Version 3), we have to synchronize PyTorch and numba explicitly.
def as_cuda_array(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.devices.get_context(t.device.index)
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'), 
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)

class render_nm(torch.autograd.Function):
    # envmap : BS * 3 * He * We
    @staticmethod
    def compute_cdf(envmap, size=(64,128)):
        I = torch.sum(envmap,dim=1,keepdim=True)
        I = nn.functional.interpolate(I, size=size, mode='area')[:,0,:,:]

        v, u = torch.meshgrid([torch.arange(0,size[0]), torch.arange(0,size[1])])
        phi = (u * 2.0 * np.pi / size[1])[None,:,:].float().to(envmap.device)
        theta = (v * np.pi / size[0])[None,:,:].float().to(envmap.device)

        # normalize pdf    
        dw = 2*np.pi/size[1]*(torch.cos(theta) - torch.cos(theta + np.pi/size[0]))
        #print(torch.sum(dw)/4/envmap.size(0)) # must be pi
        pdf = I / torch.sum(I * dw, dim=(1,2), keepdim=True) # p(w)
        p = pdf * dw # p(x,y) (x,y : normalized image coordinate)    

        cdf1 = torch.cumsum(torch.sum(p,dim=2), dim=1) # BS*H 
        p += 1e-12 * (torch.sum(p, dim=2, keepdim=True) == 0.0) # avoid zero division
        cdf2 = torch.cumsum(p, dim=2) / torch.sum(p, dim=2, keepdim=True) # phi direction
        return pdf, cdf1, cdf2

    # normal : BS * 3 * H * W
    # view   : BS * 3 * H * W
    # brdf   : BS * 3 * D * D * D
    # envmap : BS * 3 * He * We
    # output : BS * 3 * H * W
    @staticmethod
    def forward(ctx, normal, view, brdf, envmap, seed, spp):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        BS = normal.size(0)
        H = normal.size(2)
        W = normal.size(3)        
        He = envmap.size(2)
        We = envmap.size(3)        

        device = brdf.device
        brdf_weighted = torch.mean(brdf.detach(), dim=1) * merl_weight.to(device)[None]
        brdf_weighted -= torch.sum(brdf_weighted * lambert_basis[None].to(device), dim=(1,2,3), keepdim=True) * lambert_basis[None].to(device)
        l = torch.sqrt(torch.sum(brdf_weighted**2, dim=(1,2,3))) # [BS]
        dp = torch.sum(brdf_weighted[:,None] * phong_basis.to(device)[None,:], dim=(2,3,4)) # [BS,N]
        residual2 = (l[:,None]**2 - dp**2)
        alpha = phong_alphas[torch.argmin(residual2, dim=1)].to(device) # [BS]

        normal_n = as_cuda_array(normal.detach())
        view_n = as_cuda_array(view.detach())
        brdf_n = as_cuda_array(brdf.detach())
        envmap_n = as_cuda_array(envmap.detach())

        result = torch.zeros((BS,3,H,W), dtype=normal.dtype, device=normal.device)
        result_n = as_cuda_array(result)

        grad_normal = torch.zeros((BS,3,3,H,W), dtype=normal.dtype, device=normal.device)
        grad_normal_n = as_cuda_array(grad_normal)

        num_sample = int(spp.item())

        envmap_pdf, envmap_cdf1, envmap_cdf2 = render_nm.compute_cdf(envmap.detach(), size=(64,128))
        envmap_pdf_n = as_cuda_array(envmap_pdf.detach())
        envmap_cdf1_n = as_cuda_array(envmap_cdf1.detach())
        envmap_cdf2_n = as_cuda_array(envmap_cdf2.detach())
        alpha_n = as_cuda_array(alpha.detach())

        num_threads = min(256, num_sample)
        render_forward_kernel[(BS,H,W),(num_threads)](
            normal_n, view_n, brdf_n, envmap_n, 
            envmap_pdf_n, envmap_cdf1_n, envmap_cdf2_n, alpha_n, num_sample, seed.item(),
            result_n, grad_normal_n
        )

        ctx.save_for_backward(
            normal, view, brdf, envmap, 
            grad_normal, 
            spp, seed, envmap_pdf, envmap_cdf1, envmap_cdf2, alpha
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        normal, view, brdf, envmap, grad_normal,spp,seed,envmap_pdf, envmap_cdf1, envmap_cdf2,alpha, = ctx.saved_tensors
        grad_normal = matmul(grad_output[:,None,:,:,:], grad_normal, 1, 2)[:,0]
        
        if (not envmap.requires_grad) and (not brdf.requires_grad):
            return grad_normal, None, None, None, None

        BS = normal.size(0)
        H = normal.size(2)
        W = normal.size(3)        
        He = envmap.size(2)
        We = envmap.size(3)        

        normal_n = as_cuda_array(normal.detach())
        view_n = as_cuda_array(view.detach())
        brdf_n = as_cuda_array(brdf.detach())
        envmap_n = as_cuda_array(envmap.detach())

        grad_output_n = as_cuda_array(grad_output.detach())

        grad_brdf = torch.zeros_like(brdf, dtype=normal.dtype, device=normal.device)
        grad_brdf_n = as_cuda_array(grad_brdf.detach())
        grad_envmap = torch.zeros_like(envmap, dtype=normal.dtype, device=normal.device)
        grad_envmap_n = as_cuda_array(grad_envmap.detach())

        requires_grad_brdf = brdf.requires_grad
        requires_grad_envmap = envmap.requires_grad

        num_sample = int(spp.item())
        envmap_pdf_n = as_cuda_array(envmap_pdf.detach())
        envmap_cdf1_n = as_cuda_array(envmap_cdf1.detach())
        envmap_cdf2_n = as_cuda_array(envmap_cdf2.detach())
        alpha_n = as_cuda_array(alpha.detach())

        num_threads = min(256, num_sample)
        render_backward_kernel[(BS,H,W),(num_threads)](
            normal_n, view_n, brdf_n, envmap_n, 
            envmap_pdf_n, envmap_cdf1_n, envmap_cdf2_n, alpha_n, num_sample, seed.item(),
            grad_output_n, grad_brdf_n, grad_envmap_n, 
            requires_grad_brdf, requires_grad_envmap
        )

        return grad_normal, None, grad_brdf, grad_envmap, None, None

class render_nm_bf(torch.autograd.Function):
    # normal : BS * 3 * H * W
    # view   : BS * 3 * H * W
    # brdf   : BS * 3 * D * D * D
    # envmap : BS * 3 * He * We
    # output : BS * 3 * H * W
    @staticmethod
    def forward(ctx, normal, view, brdf, envmap, seed, spp):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        BS = normal.size(0)
        H = normal.size(2)
        W = normal.size(3)        
        He = envmap.size(2)
        We = envmap.size(3)

        normal_n = as_cuda_array(normal.detach())
        view_n = as_cuda_array(view.detach())
        brdf_n = as_cuda_array(brdf.detach())
        envmap_n = as_cuda_array(envmap.detach())
        
        result = torch.zeros((BS,3,H,W), dtype=normal.dtype, device=normal.device)
        result_n = as_cuda_array(result)

        grad_normal = torch.zeros((BS,3,3,H,W), dtype=normal.dtype, device=normal.device)
        grad_normal_n = as_cuda_array(grad_normal)  

        num_threads = min(256, He)
        render_forward_bf_kernel[(BS,H,W),(num_threads)](normal_n, view_n, brdf_n, envmap_n, result_n, grad_normal_n)
        ctx.save_for_backward(normal, view, brdf, envmap, grad_normal, seed)
        return result


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        normal, view, brdf, envmap, grad_normal, seed, = ctx.saved_tensors
        grad_normal = matmul(grad_output[:,None,:,:,:], grad_normal, 1, 2)[:,0]
        
        if (not envmap.requires_grad) and (not brdf.requires_grad):
            return grad_normal, None, None, None, None

        BS = normal.size(0)
        H = normal.size(2)
        W = normal.size(3)        
        He = envmap.size(2)
        We = envmap.size(3)        

        normal_n = as_cuda_array(normal.detach())
        view_n = as_cuda_array(view.detach())
        brdf_n = as_cuda_array(brdf.detach())
        envmap_n = as_cuda_array(envmap.detach())

        grad_output_n = as_cuda_array(grad_output.detach())

        grad_brdf = torch.zeros_like(brdf, dtype=normal.dtype, device=normal.device)
        grad_brdf_n = as_cuda_array(grad_brdf.detach())
        grad_envmap = torch.zeros_like(envmap, dtype=normal.dtype, device=normal.device)
        grad_envmap_n = as_cuda_array(grad_envmap.detach())

        requires_grad_brdf = brdf.requires_grad
        requires_grad_envmap = envmap.requires_grad

        num_threads = min(256, He)
        render_backward_bf_kernel[(BS,H,W),(num_threads)](normal_n, view_n, brdf_n, envmap_n, grad_output_n, grad_brdf_n, grad_envmap_n, requires_grad_brdf, requires_grad_envmap)
        return grad_normal, None, grad_brdf, grad_envmap, None, None


class Renderer(torch.nn.Module):
    def __init__(self, use_importance_sampling=False, initial_seed=0):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Renderer, self).__init__()
        if use_importance_sampling:
            self.render = render_nm.apply
        else:
            self.render = render_nm_bf.apply
        self.seed=initial_seed

    def forward(self,  normal, view, brdf, envmap, seed=None, spp=1024):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # zenith = z axis -> zenith = y axis
        R = torch.tensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, -1.0], 
            [0.0, 1.0, 0.0], 
        ], dtype=normal.dtype, device=normal.device)
        normal = matmul(R[None,:,:,None,None], normal[:,:,None,:,:], 1, 2)[:,:,0]
        view = matmul(R[None,:,:,None,None], view[:,:,None,:,:], 1, 2)[:,:,0]
        if seed is None:
            seed = torch.tensor(self.seed, dtype=torch.int64, device=normal.device)
            if self.training:
                self.seed += np.array(normal[:,0,:,:].size()).prod() * spp
        y = self.render(normal, view, brdf, envmap, seed, torch.tensor(spp))
        return y
    
    # normal: BS,3,H,W # [right,up,-lookat]
    # brdf:   BS,3,90,90,180
    # envmap: BS,3,He,We
    # extrinsic: BS,4,4    
    def render_orthographic(self, normal, brdf, envmap, extrinsic, num_itr=1, spp=1024):
        BS,C,H,W = normal.size()
        # to global normal
        rot = extrinsic[:,:3,:3]
        normal = normal * torch.tensor([1.0,-1.0,-1.0], dtype=normal.dtype, device=normal.device)[None,:,None,None]
        normal = matmul(rot.transpose(1,2)[:,:,:,None,None], normal[:,:,None,:,:], 1, 2)[:,:,0]
        
        # compute viewing direction
        view = -rot[:,2,:,None,None].repeat(1,1,H,W)
        
        x = 1.0 / num_itr * self.forward(normal, view, brdf, envmap, spp=spp)
        for i in range(num_itr-1):
            x = x + 1.0 / num_itr * self.forward(normal, view, brdf, envmap)
        return x