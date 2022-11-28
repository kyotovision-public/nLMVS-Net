import os
import numpy as np
import torch

from .ibrdf.util import *

reference = LoadMERL('./data/alum-bronze.binary').float()

def generate_ibrdf_grid(unwarp=True):

    # make grid
    with torch.no_grad():
        mask = (reference[0] > 0.0).to(torch.int64)

    thetaH = torch.linspace(0.5 / BRDFSamplingResThetaH, 1.0 - 0.5 / BRDFSamplingResThetaH, BRDFSamplingResThetaH)
    thetaD = torch.linspace(0.5 / BRDFSamplingResThetaD, 1.0 - 0.5 / BRDFSamplingResThetaD, BRDFSamplingResThetaD)
    phiD = torch.linspace(1.0 / BRDFSamplingResPhiD, 1.0 - 1.0 / BRDFSamplingResPhiD, BRDFSamplingResPhiD // 2)

    axisTuple = torch.meshgrid((thetaH, thetaD, phiD))
    grid = torch.stack((axisTuple[0].flatten(), axisTuple[1].flatten(), axisTuple[2].flatten()), dim=1)
    grid = grid.view(BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD // 2, 3)
    grid = grid
    if unwarp:
        bound = LastNonZero(mask)
        scaleRatio = (bound.to(torch.float32) + 1.0) / BRDFSamplingResThetaH
        grid[...,0] = grid[...,0] / scaleRatio
        grid[grid >= 1.0] = 0.0
    grid = grid.view(-1,3)

    return grid, mask

def decode_brdf(ibrdf_model, embed_code, log_color):
    device = embed_code.device
    ibrdf_grid, ibrdf_mask = generate_ibrdf_grid()
    ibrdf_grid = ibrdf_grid.to(device)
    ibrdf_mask = ibrdf_mask.to(device)

    brdf = []
    for idx_b in range(embed_code.size(0)):
        brdf_per_instance = []
        for idx_l in range(embed_code.size(1)):
            z = embed_code[idx_b,idx_l:idx_l+1].view(-1,16).repeat(ibrdf_grid.size(0),1)
            lobe = ibrdf_model.logPDF(ibrdf_grid, z).exp().view(90,90,180) * ibrdf_mask
            lobe = lobe / torch.max(lobe) * np.log1p(10000.0)
            brdf_ = torch.expm1(lobe) * log_color[idx_b,idx_l].exp()
            brdf_per_instance.append(brdf_)
        brdf_per_instance = torch.stack(brdf_per_instance, dim=0)
        brdf.append(brdf_per_instance)

    brdf = torch.stack(brdf, dim=0)
    brdf = brdf / torch.tensor([1.0, 1.15, 1.66])[None,:,None,None,None].to(device) * 1.241
    return brdf

# embed_code: [BS,3,16]
# log_color: [BS,3]
# grad_brdf: [BS,3,90,90,180]
def backward_brdf(ibrdf_model, embed_code, log_color, grad_brdf, ChunkSize=150000):
    device = embed_code.device
    ibrdf_grid, ibrdf_mask = generate_ibrdf_grid()
    ibrdf_grid = ibrdf_grid.to(device)
    ibrdf_mask = ibrdf_mask.to(device)

    # generate lobes
    lobes = []
    for idx_b in range(embed_code.size(0)):
        lobes_per_instance = []
        for idx_l in range(embed_code.size(1)):
            with torch.no_grad():
                z = embed_code[idx_b,idx_l:idx_l+1].view(-1,16).repeat(ibrdf_grid.size(0),1)
                lobe = ibrdf_model.logPDF(ibrdf_grid, z).exp().view(90,90,180) * ibrdf_mask
            lobes_per_instance.append(lobe)
        lobes_per_instance = torch.stack(lobes_per_instance, dim=0)
        lobes.append(lobes_per_instance)
    lobes = torch.stack(lobes, dim=0) # [BS,3,90,90,180]
    with torch.no_grad():
        lobes.requires_grad = True
    
    brdf = []
    for idx_b in range(embed_code.size(0)):
        brdf_per_instance = []
        for idx_l in range(embed_code.size(1)):
            lobe = lobes[idx_b,idx_l]
            lobe = lobe / torch.max(lobe) * np.log1p(10000.0)
            brdf_ = torch.expm1(lobe) * log_color[idx_b,idx_l].exp()
            brdf_per_instance.append(brdf_)
        brdf_per_instance = torch.stack(brdf_per_instance, dim=0)
        brdf.append(brdf_per_instance)
    brdf = torch.stack(brdf, dim=0) # [BS,3,90,90,180]
    brdf = brdf / torch.tensor([1.0, 1.15, 1.66])[None,:,None,None,None].to(device) * 1.241

    # backward to lobes and log_color
    brdf.backward(grad_brdf)

    with torch.no_grad():
        grad_lobes = lobes.grad
        nonZeroGradsIndex = torch.nonzero(grad_lobes)

    for chunkIdx in range((len(nonZeroGradsIndex) - 1) // ChunkSize + 1):
        nonZeroGradsIndexChunk = nonZeroGradsIndex[ChunkSize*chunkIdx:ChunkSize*(chunkIdx+1)] # [N,5]
        idx_b = nonZeroGradsIndexChunk[:,0]
        idx_c = nonZeroGradsIndexChunk[:,1]

        positions = nonZeroGradsIndexChunk[:,2:5]
        embed_code_chunk = embed_code[idx_b, idx_c]

        lobes_subset = generateMERLSamples(ibrdf_model, positions, reference, embed_code_chunk)
        
        grad_lobes_subset = grad_lobes[idx_b,idx_c,positions[:,0],positions[:,1],positions[:,2]]
        lobes_subset.backward(grad_lobes_subset)

def render_sphere(renderer, brdf, illum_map, extrinsic, resolution=128, spp=4096):
    device = brdf.device
    # compute normal map of sphere
    ny, nx = torch.meshgrid(torch.arange(resolution), torch.arange(resolution))
    ny = -(2 * (ny.float().to(device) + 0.5) / resolution - 1.0)
    nx = 2 * (nx.float().to(device) + 0.5) / resolution - 1.0
    nz = torch.sqrt(torch.clamp(1 - nx**2 - ny**2, 0, 1))
    rmap_normal = torch.stack([nx, ny, nz], dim=0)[None] # 1,3,Hn,Wn
    rmap_normal *= (nz > 0.0).float()[None,None]

    return renderer.render_orthographic(rmap_normal, brdf, illum_map, extrinsic, spp=spp)