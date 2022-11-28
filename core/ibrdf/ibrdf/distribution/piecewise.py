import numpy as np

import torch
import torch.nn as nn

import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

# arr: [N]
# val: scalar
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

@cuda.jit
def sample_kernel(cdf1, cdf2, cdf3, z, result):
    #N, Nx, Ny, Nz = cdf3.shape
    num_samples = z.shape[1]

    idx_brdf = cuda.blockIdx.x
    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    idx_sample = idx_thread
    while idx_sample < num_samples:
        rand_x, rand_y, rand_z = z[idx_brdf, idx_sample]
        idx_x = binsearch_device(cdf1[idx_brdf], rand_x)
        idx_y = binsearch_device(cdf2[idx_brdf,idx_x], rand_y)
        idx_z = binsearch_device(cdf3[idx_brdf,idx_x,idx_y], rand_z)

        result[idx_brdf,idx_sample,0] = idx_x
        result[idx_brdf,idx_sample,1] = idx_y
        result[idx_brdf,idx_sample,2] = idx_z

        idx_sample += blockdim


# brdf: [N,90,90,180]
# num_samples: number of samples per brdf
def sample_coords(brdf, num_samples):
    dtype = brdf.dtype
    device = brdf.device
    N,Nx,Ny,Nz = brdf.size()
    cdf1 = torch.cumsum(torch.sum(brdf, dim=(2,3)), dim=1)
    cdf1 /= torch.clamp(cdf1[:,-1:], 1e-12, None) # [N,90]
    cdf2 = torch.cumsum(torch.sum(brdf, dim=3), dim=2)
    cdf2 /= torch.clamp(cdf2[:,:,-1:], 1e-12, None) # [N,90,90]
    cdf3 = torch.cumsum(brdf, dim=3)
    cdf3 /= torch.clamp(cdf3[:,:,:,-1:], 1e-12, None) # [N,90,90,180]

    z = torch.rand((N,num_samples,3), dtype=dtype, device=device)
    result = torch.zeros_like(z, dtype=torch.int64, device=device)

    cdf1_n = cuda.as_cuda_array(cdf1.detach())
    cdf2_n = cuda.as_cuda_array(cdf2)
    cdf3_n = cuda.as_cuda_array(cdf3)
    z_n = cuda.as_cuda_array(z)
    result_n = cuda.as_cuda_array(result)

    num_threads = min(256, num_samples)
    sample_kernel[(N,),(num_threads,)](cdf1_n, cdf2_n, cdf3_n, z_n, result_n)

    result = result.to(dtype) + torch.rand_like(result, dtype=dtype, device=device)
    
    result = result / torch.tensor([Nx,Ny,Nz], dtype=dtype, device=device)[None,None,:]

    return result
