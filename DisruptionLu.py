import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from DataCollection import *
import random
import csv
from Solver_vote import *
import torch
from LuNet_vote import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

lr = 1e-1
max_norm = 1.0
max_epoch = 10
break_loss = 0.001

net = LuNet_vote()
# net.load_state_dict(torch.load('./LuNetf.pkl'))
net = net.to('cuda')
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
# optimizer = optimizer.to('cuda')
criterion = torch.nn.BCELoss()
criterion = criterion.to('cuda')
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,30,50,70,90,110,130,150],gamma=0.2)
# scheduler = scheduler.to('cuda')
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,30,50,70,90,110,130,150],gamma=0.1)



train_shotlist,validate_shotlist = Generate_Batch()
Training_Loss,validate_Loss,T_accu,V_accu = Train_and_Validate(net,optimizer,scheduler,criterion,max_norm,train_shotlist,validate_shotlist,max_epoch,break_loss)

plt.plot(Training_Loss,label='Train')
plt.title('train')
plt.show()
plt.plot(validate_Loss,label = 'Validate')
plt.title('validate')
plt.show()
plt.plot(T_accu)
plt.title('train accu')
plt.show()
plt.plot(V_accu)
plt.title('validate accu')
plt.show()
message = input('Save(1) or not(0):')
if(message):
    net = net.to('cpu')
    torch.save(net.state_dict(),'./LuNetf.pkl')