import numpy as np
import torch
import torch.nn as nn
from LuNet7_dy import *
from DataCollection import *

def Train_and_Validate(net,optimizer,scheduler,criterion,train_shotlist,validate_shotlist,max_epoch):
    train_setsize = len(train_shotlist)
    validate_setsize = len(validate_shotlist)
    # train_set = torch.tensor(train_set)
    # train_label = torch.tensor(train_label)
    # validate_set = torch.tensor(validate_set)
    # validate_label = torch.tensor(validate_label)
    Training_Loss = []
    Validate_Loss = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(max_epoch):
        correct = []
        Training_Loss_Epoch = []
        for i in range(train_setsize):
            a, isdisrupt,_ = Get_signal_from_shot(train_shotlist[i],4)
            a = Size_Normalization(a)
            datain = torch.tensor(a).reshape(1,-1).to(torch.float32).to('cuda')
            label = torch.tensor(isdisrupt).reshape(1).to(torch.float32).to('cuda')
            # print(datain.size())
            output = net(datain).reshape(1)
            print(output,label)
            loss = criterion(output,label)
            # print(loss)
            loss.backward()
            # print(loss)
            optimizer.step()
            # print(loss)
            average_loss = loss.item()
            prediction = torch.round(output.detach())
            gt = label.detach()
            correct.append(torch.eq(prediction,gt).item())
            # print(correct)
            # Training_Loss.append(average_loss)
            Training_Loss_Epoch.append(average_loss)
            # print(average_loss)
        correct = torch.tensor(correct)
        accu = torch.sum(correct)/train_setsize
        Training_Loss_Epoch = np.mean(Training_Loss_Epoch)
        Training_Loss.append(Training_Loss_Epoch)
        print('Train: Epoch [%u]\t Loss[%.04f]\t Accu[%.04f]'%(epoch, Training_Loss_Epoch,accu))
        scheduler.step()
        correct = []
        Validate_Loss_Epoch = []
        for i in range(validate_setsize):
            a, isdisrupt,_ = Get_signal_from_shot(validate_shotlist[i],4)
            a = Size_Normalization(a)
            datain = torch.tensor(a).reshape(1,-1).to(torch.float32).to('cuda')
            label = torch.tensor(isdisrupt).reshape(1).to(torch.float32).to('cuda')
            output = net(datain).reshape(1)
            loss = criterion(output,label)
            average_loss = loss.item()
            prediction = torch.round(output.detach())
            gt = label.detach()
            correct.append(torch.eq(prediction,gt).item())
            Validate_Loss_Epoch.append(average_loss)
        correct = torch.tensor(correct)
        accu = torch.sum(correct)/validate_setsize
        Validate_Loss_Epoch = np.mean(Validate_Loss_Epoch)
        print('Validate: Epoch [%u]\t Loss[%.04f]\t Accu[%.04f]'%(epoch, Validate_Loss_Epoch,accu))
        Validate_Loss.append(Validate_Loss_Epoch)
    return Training_Loss,Validate_Loss