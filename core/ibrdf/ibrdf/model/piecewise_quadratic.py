import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .coupling_layer import CouplingLayer
from .unet import UNet

class PiecewiseQuadraticCoupling(CouplingLayer):
    def __init__(self, inFeatures, mask, k=32, uNetBins=1, numEmbedDim=0):
        super(PiecewiseQuadraticCoupling, self).__init__(mask)
        self.K = k

        uNetIn = int(mask.sum().item()) + numEmbedDim
        uNetOut = (inFeatures - int(mask.sum().item())) * (k + k + 1)

        self.netM = UNet(uNetIn, uNetOut, uNetBins)

    def getParams(self, xa, xb, embedCode):
        if not (embedCode is None):
            netIn = torch.cat([xa, embedCode], -1)
        else:
            netIn = xa

        params = self.netM(netIn)
        params = params.view(params.size(0), xb.size(1), 2 * self.K + 1)

        vw = params.split((self.K + 1, self.K), -1)
        w = torch.softmax(vw[1], -1)
        vExp = vw[0].exp()
        vNormalizer = ((vExp.roll(1, -1) + vExp)[..., 1:] / 2.0 * w).sum(-1, True)
        v = vExp / vNormalizer
        
        return v, w

    def coupleForward(self, xa, xb, embedCode):
        v, w = self.getParams(xa, xb, embedCode)

        wCumSum = w.roll(1, -1)
        wCumSum[..., 0] = 0.0
        wCumSum = wCumSum.to(torch.float64).cumsum(-1).to(w.dtype)

        b = torch.searchsorted(wCumSum.detach().squeeze(1), xb.detach(), right=True).unsqueeze(-1) - 1

        a = xb - wCumSum.gather(-1, b).squeeze(-1)
        aNormalized = a / w.gather(-1, b).squeeze(-1)

        vw = ((v + v.roll(1, -1))[...,1:] / 2.0 * w).roll(1, -1)
        vw[..., 0] = 0.0
        vwCumSum = vw.to(torch.float64).cumsum(-1).to(vw.dtype)

        vIb1 = v.gather(-1, b + 1).squeeze(-1)
        vIb = v.gather(-1, b).squeeze(-1)

        yb = a * a / (2.0 * w.gather(-1, b).squeeze(-1)) * (vIb1 - vIb) + a * vIb + vwCumSum.gather(-1, b).squeeze(-1)

        # Workaround rounding errors
        yb = yb.masked_scatter((yb >= 1.0), yb - 1e-6)

        det = vIb + aNormalized * (vIb1 - vIb)
        logDet = det.log().sum(-1)

        return yb, logDet

    def coupleInverse(self, ya, yb, embedCode):
        v, w = self.getParams(ya, yb, embedCode)

        vw = ((v + v.roll(1, -1))[..., 1:] / 2.0 * w).roll(1, -1)
        vw[..., 0] = 0.0
        vwCumSum = vw.to(torch.float64).cumsum(-1).to(vw.dtype)

        b = torch.searchsorted(vwCumSum.detach().squeeze(1), yb.detach(), right=True).unsqueeze(-1) - 1

        vIb1 = v.gather(-1, b + 1).squeeze(-1)
        vIb = v.gather(-1, b).squeeze(-1)

        eqA = (vIb1 - vIb) / (2.0 * w.gather(-1, b).squeeze(-1))
        eqB = vIb
        eqC = vwCumSum.gather(-1, b).squeeze(-1) - yb

        disc = (eqB * eqB - 4.0 * eqA * eqC).sqrt()
        # Ref: https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        eqSol1 = (-eqB - disc) / (2.0 * eqA)
        eqSol2 = (2.0 * eqC) / (-eqB - disc)

        # * is used here instead of element-wise and.
        a = torch.where((eqSol1 > 0.0) * (eqSol1.abs() < eqSol2.abs()), eqSol1, eqSol2)
        aNormalized = a / w.gather(-1, b).squeeze(-1)

        wCumSum = w.roll(1, -1)
        wCumSum[..., 0] = 0.0
        wCumSum = wCumSum.to(torch.float64).cumsum(-1).to(w.dtype)

        xb = a + wCumSum.gather(-1, b).squeeze(-1)

        # Workaround rounding errors
        xb = xb.masked_scatter(xb >= 1.0, xb - 1e-6)

        det = vIb + aNormalized * (vIb1 - vIb)
        logDet = -det.log().sum(-1)

        return xb, logDet

    def couple(self, xa, xb, embedCode, reverse):
        if reverse:
            return self.coupleInverse(xa, xb, embedCode)
        else:
            return self.coupleForward(xa, xb, embedCode)