import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CouplingLayer(nn.Module):
    def __init__(self, mask):
        super(CouplingLayer, self).__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('maskInv', torch.logical_not(mask))

    # must be overridden
    def couple(self, xa, xb, embedCode, reverse):
        print('Function couple() of CoupplingLayer class is called!!')
        print('This function must be overridden!!')
        return 0

    def forward(self, x, embedCode=None, reverse=False):
        xa = torch.masked_select(x, self.mask).view(x.size(0),-1)
        xb = torch.masked_select(x, self.maskInv).view(x.size(0),-1)
        yb, det = self.couple(xa, xb, embedCode, reverse)
        y = x.masked_scatter(self.maskInv, yb)

        return y, det
