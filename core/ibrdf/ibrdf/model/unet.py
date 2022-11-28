import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .util import oneBlob

class UNet(nn.Module):
    def __init__(self, inFeatures, outFeatures, numBins):
        super(UNet, self).__init__()

        self.numBins = numBins
        self.pre = nn.Linear(inFeatures*numBins, 256)
        self.e1 = nn.Linear(256, 128)
        self.e2 = nn.Linear(128, 64)
        self.e3 = nn.Linear(64, 32)
        self.e4 = nn.Linear(32, 16)

        self.d1 = nn.Linear(16, 32)
        self.d2 = nn.Linear(64, 64)
        self.d3 = nn.Linear(128, 128) 
        self.d4 = nn.Linear(256, 256)
        self.post = nn.Linear(256, outFeatures)

    def forward(self, x):
        if self.numBins > 1:
            x = oneBlob(x, self.numBins)

        x = self.pre(x)

        h1 = self.e1(F.relu(x))
        h2 = self.e2(F.relu(h1))
        h3 = self.e3(F.relu(h2))
        h4 = self.e4(F.relu(h3))

        y = self.d1(F.relu(h4))
        y = self.d2(F.relu(torch.cat([y, h3], dim=1)))
        y = self.d3(F.relu(torch.cat([y, h2], dim=1)))
        y = self.d4(F.relu(torch.cat([y, h1], dim=1)))

        y = self.post(F.relu(y))

        return y