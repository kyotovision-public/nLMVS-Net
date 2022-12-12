import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .piecewise_quadratic import PiecewiseQuadraticCoupling


class IBRDF(nn.Module):
    def __init__(self, numLayers, inFeatures, numEmbedDims=0, numPieces=32, numUnetBins=1, transform = 'quadratic'):
        super(IBRDF, self).__init__()
        self.inFeatures = inFeatures

        layers = []
        for i in range(numLayers):
            mask = torch.ones((inFeatures), dtype=torch.bool)
            mask[i % inFeatures] = 0
            if transform == 'quadratic':
                layers.append(PiecewiseQuadraticCoupling(inFeatures, mask, numPieces, numUnetBins, numEmbedDims))
            else:
                print('transform type \''+transform+'\' is not implemented!')
                exit(1)
        self.layers = nn.ModuleList(layers)



    def forward(self, x, embedCode=None, reverse = False):
        result = x
        logDet = torch.zeros((x.size(0)), device=x.device, dtype=x.dtype)
        if reverse:
            for i in range(len(self.layers) - 1, -1, -1):
                x_, logDet_ = self.layers[i](result, embedCode, reverse)
                result = x_
                logDet = logDet + logDet_
        else:
            for layer in self.layers:
                x_, logDet_ = layer(result, embedCode, reverse)
                result = x_
                logDet = logDet + logDet_
        return result, logDet

    def sample(self, numSamples, embedCode=None):
        for p in self.parameters():
            break
        device = p.device
        z = torch.rand((numSamples, self.inFeatures), device=device)
        x = self.forward(z, embedCode, True)[0]
        
        return x

    def logPDF(self, x, embedCode=None):
        return self.forward(x, embedCode)[-1]