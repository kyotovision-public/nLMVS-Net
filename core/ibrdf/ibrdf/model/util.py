import torch
import numpy as np

def oneBlob(x, numBins):
    grid = torch.linspace(0.0, 1.0, numBins, dtype=x.dtype, device=x.device)

    sigma = 1.0 / numBins
    normalizer = 1.0 / np.sqrt(2.0 * np.pi * sigma * sigma)

    mean = x.unsqueeze(-1).repeat(1,1,numBins)

    return normalizer * torch.exp(-(grid - mean)**2 / (2.0 * sigma * sigma)).view(x.size(0), -1)