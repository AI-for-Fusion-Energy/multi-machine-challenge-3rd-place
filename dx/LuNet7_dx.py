import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LuNet7_dx(nn.Module):
    def __init__(self):
        super(LuNet7_dx,self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1,4,9,padding='valid')
        self.conv2 = nn.Conv1d(4,4,8,padding='valid')
        self.fc = nn.Linear(4,1)
    
    def forward(self,X):
        X = F.avg_pool1d(X,125)
        X = X.reshape(1,1,-1)
        X = self.bn(X)
        X = F.leaky_relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        X = X.reshape(1,-1)
        X = F.sigmoid(self.fc(X))
        return X