import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LuNet_vote(nn.Module):
    def __init__(self):
        super(LuNet_vote,self).__init__()
        # self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(9,3)

    def forward(self,X):
        X = F.max_pool1d(self.fc(X).reshape(1,3),3)
        X = F.sigmoid(X).reshape(1)
        return X