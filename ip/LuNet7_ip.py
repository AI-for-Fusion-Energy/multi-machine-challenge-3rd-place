import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LuNet7_ip(nn.Module):
    def __init__(self):
        super(LuNet7_ip,self).__init__()
        # self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1,4,3,padding='valid',bias = False)
        self.conv2 = nn.Conv1d(4,4,3,padding='valid',bias = False)
        self.fc = nn.Linear(12,1)
    
    def forward(self,X):
        X = F.avg_pool1d(X,125)
        # plt.plot(X[0].detach().to('cpu'))
        # plt.show()
        X = F.leaky_relu(self.conv1(X))
        # plt.plot(X[0].detach().to('cpu'))
        # plt.show()
        # print(X[16])
        # X = F.max_pool1d(X,7)
        
        X = self.conv2(X)
        X = X.transpose(0,1)
        X = F.max_pool1d(X,4).squeeze()
        X = self.fc(X)
        # print(X)
        X = F.sigmoid(X)
        return X