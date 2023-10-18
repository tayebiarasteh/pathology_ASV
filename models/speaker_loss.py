"""
Generalized end-to-end loss for speaker verification.
Created on Wed Sep  5 20:58:34 2018

@author: Harry Volek
https://github.com/HarryVolek/PyTorch_Speaker_Verification
"""

import torch.nn as nn
import torch

from utils.utils import *


class GE2ELoss(nn.Module):

    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.setup_cuda()
        # w and b are learnable parameters with initial values of 10 and -5
        self.w = nn.Parameter(torch.tensor(10.0).to(self.device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(self.device), requires_grad=True)

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings) # (N, 256)
        cossim = get_cossim(embeddings, centroids) # (N, M, N)
        sim_matrix = self.w * cossim.to(self.device) + self.b # (N, M, N)
        loss, _ = calc_loss(sim_matrix)
        return loss

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
