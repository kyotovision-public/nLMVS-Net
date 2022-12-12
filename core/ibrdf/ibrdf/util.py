import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import struct

BRDFSamplingResThetaH = 90
BRDFSamplingResThetaD = 90
BRDFSamplingResPhiD = 360

RedScale = 1.0 / 1500.0
GreenScale = 1.15 / 1500.0
BlueScale = 1.66 / 1500.0

def LastNonZero(t):
    if t.ndim != 3:
        raise Exception("Input tensor must be 3 dimension!")

    result = torch.argmax((t == 0.0).float(), dim=0) - 1
    result[result == -1] = t.size(0) - 1
    return result

def LoadMERL(path):
    with open(path, 'rb') as f:
        bdata = f.read()
    dims = []
    dims.append(int.from_bytes(bdata[:4], 'little'))
    dims.append(int.from_bytes(bdata[4:8], 'little'))
    dims.append(int.from_bytes(bdata[8:12], 'little'))
    n = np.prod(dims)
    if n != (BRDFSamplingResThetaH * BRDFSamplingResThetaD * BRDFSamplingResPhiD / 2):
        raise Exception('Dimensions don\'t match')
    brdf = struct.unpack(str(3*n)+'d', bdata[12:])

    brdf = np.reshape(np.array(brdf), (3,BRDFSamplingResThetaH,BRDFSamplingResThetaD,BRDFSamplingResPhiD//2))
    brdf = torch.from_numpy(brdf.astype(np.float))
    return brdf

def SaveMERL(path, material):
    m = material.detach().cpu().flatten().contiguous().numpy()

    with open(path, 'wb') as f:
        dims = [BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD//2]
        for dim in dims:
            f.write(dim.to_bytes(4, 'little'))
        
        f.write(struct.pack('<{}d'.format(len(m)), *m))

def generateMERLSamples(model, positions, reference, embedCode, unwarp=True):
    for p in model.parameters():
        break
    device = p.device

    with torch.no_grad():
        pos = positions.cpu() / torch.tensor([BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD / 2]).float() 
        pos += torch.tensor([0.5 / BRDFSamplingResThetaH, 0.5 / BRDFSamplingResThetaD, 0.5 / (BRDFSamplingResPhiD / 2)]).float()

        mask = (reference[0] > 0.0).to(torch.int64).cpu()
        maskSubset = mask[positions[:,0], positions[:,1], positions[:,2]]

        if unwarp:
            bound = LastNonZero(mask)
            scaleRatio = (bound.to(torch.float32) + 1.0) / BRDFSamplingResThetaH
            scaleRatioSubset = scaleRatio[positions[:,1], positions[:,2]]
            pos[..., 0] = pos[..., 0] / scaleRatioSubset
            pos[pos >= 1.0] = 0.0

    pos = pos.to(device)

    return model.logPDF(pos, embedCode).exp().squeeze(-1) * maskSubset.to(device)



def generateMERLSlice(model, reference, embedCode=None, unwarp=True):
    for p in model.parameters():
        break
    device = p.device

    with torch.no_grad():
        mask = (reference[0] > 0.0).to(torch.int64).to(device)

    thetaH = torch.linspace(0.5 / BRDFSamplingResThetaH, 1.0 - 0.5 / BRDFSamplingResThetaH, BRDFSamplingResThetaH)
    thetaD = torch.linspace(0.5 / BRDFSamplingResThetaD, 1.0 - 0.5 / BRDFSamplingResThetaD, BRDFSamplingResThetaD)
    phiD = torch.linspace(1.0 / BRDFSamplingResPhiD, 1.0 - 1.0 / BRDFSamplingResPhiD, BRDFSamplingResPhiD // 2)

    axisTuple = torch.meshgrid((thetaH, thetaD, phiD))
    grid = torch.stack((axisTuple[0].flatten(), axisTuple[1].flatten(), axisTuple[2].flatten()), dim=1)
    grid = grid.view(BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD // 2, 3)
    grid = grid.to(device)
    if unwarp:
        bound = LastNonZero(mask)
        scaleRatio = (bound.to(torch.float32) + 1.0) / BRDFSamplingResThetaH
        grid[...,0] = grid[...,0] / scaleRatio
        grid[grid >= 1.0] = 0.0
    grid = grid.view(-1,3)

    return model.logPDF(grid, embedCode).exp().view(BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD // 2) * mask.to(device)

# 
def generateMERL(model, reference, embedCode, color, unwarp=True):
    for p in model.parameters():
        break
    device = p.device
    numLobes = embedCode.size(0)

    material = torch.zeros((3,90*90*180), dtype=torch.float32, device=device)

    for lobeIdx in range(numLobes):
        lobe = generateMERLSlice(model, reference, embedCode[lobeIdx].repeat(90 * 90 * 180, 1), unwarp)
        lobe = lobe / torch.max(lobe) * np.log1p(10000.0)
        lobe = torch.expm1(lobe.view(1,-1)) * color[:,lobeIdx:lobeIdx+1]
        material = material + lobe

    return material.view(3,90,90,180)

def LoadHDR(filename):
    img = cv2.imread(filename, -1)[:,:,::-1].astype(np.float)
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img)
    return img

def SaveHDR(filename, img, idx_batch=0):
    if img.ndim == 4:
        img = img[idx_batch]
    img = img.detach().cpu().numpy().transpose(1,2,0)
    cv2.imwrite(filename, img[:,:,::-1])