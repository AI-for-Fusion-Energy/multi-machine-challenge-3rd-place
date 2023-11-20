import numpy as np
import torch
import torch.nn as nn
from LuNet_vote import *
from DataCollection import *
from MagneticAnalysis import *
from test import *

def Train_and_Validate(net,optimizer,scheduler,criterion,max_norm, train_shotlist,validate_shotlist,max_epoch,break_loss):
    train_setsize = len(train_shotlist)
    validate_setsize = len(validate_shotlist)

    Training_Loss = []
    Validate_Loss = []
    Train_accu = []
    Validate_accu = []

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(max_epoch):
        Train_Loss_Epoch,T_accu = Train_vote(net,optimizer,scheduler,criterion,train_shotlist,epoch)
        Validate_Loss_Epoch,V_accu = Validate_vote(net,criterion,validate_shotlist,epoch)
        Training_Loss += Train_Loss_Epoch
        Validate_Loss += Validate_Loss_Epoch
        Train_accu.append(T_accu)
        Validate_accu.append(V_accu)
        if np.max([np.max(Train_Loss_Epoch),np.max(Validate_Loss_Epoch)])<break_loss:
            break
    return Training_Loss, Validate_Loss,Train_accu,Validate_accu

def Train_vote(net,optimizer,scheduler,criterion,train_shotlist,epoch):
    Train_Loss_Epoch = []
    signalnum_list = [1,3,4,5,8,9,89,90,91]
    correct = 0
    seq_num = len(train_shotlist)
    for file in train_shotlist:
        result_prev = []
        for signalnum in signalnum_list:
            a,label = Vote_Sample_Generator(file,signalnum)
            
            if a is None:
                result_prev.append(-1)
                continue
            
            input_data = Test_Sample_Normalize(a,signalnum)
            if isinstance(input_data,bool):
                result_prev.append(1)
                continue
            net_prev = Net_init(signalnum)
            result_prev.append(net_prev(input_data[0]).reshape(1).detach().item())
        result_prev = torch.tensor(np.array(result_prev)).to(torch.float32).to('cuda')
        output = net(result_prev)
        label = torch.tensor(np.array([label])).to(torch.float32).to('cuda')
        # print(output.shape,label.shape)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        average_loss = loss.sum().item()
        prediction = torch.round(output.detach())
        gt = label.detach()
        correct += torch.eq(prediction,gt).sum().item()
        Train_Loss_Epoch.append(average_loss)
        print('Train: Epoch [%u]\t  Loss[%.04f]'%(epoch,  average_loss))
    return Train_Loss_Epoch,correct/seq_num

def Validate_vote(net,criterion,validate_shotlist,epoch):
    Validate_Loss_Epoch = []
    signalnum_list = [1,3,4,5,8,9,89,90,91]
    correct = 0
    seq_num = len(validate_shotlist)
    for file in validate_shotlist:
        result_prev = []
        for signalnum in signalnum_list:
            a,label = Vote_Sample_Generator(file,signalnum)
            if a is None:
                result_prev.append(-1)
                continue
            
            input_data = Test_Sample_Normalize(a,signalnum)
            if isinstance(input_data,bool):
                result_prev.append(1)
                continue
            net_prev = Net_init(signalnum)
            result_prev.append(net_prev(input_data[0]).reshape(1).detach().item())
        result_prev = torch.tensor(np.array(result_prev)).to(torch.float32).to('cuda')
        output = net(result_prev)
        label = torch.tensor(np.array([label])).to(torch.float32).to('cuda')
        loss = criterion(output,label)
        average_loss = loss.sum().item()
        prediction = torch.round(output.detach())
        gt = label.detach()
        correct += torch.eq(prediction,gt).sum().item()
        Validate_Loss_Epoch.append(average_loss)
        print('Validate: Epoch [%u]\t  Loss[%.04f]'%(epoch,  average_loss))
    return Validate_Loss_Epoch,correct/seq_num

def Vote_Sample_Generator(file,signalnum):
    try:
        a,label,_ = Get_signal_from_shot(file,signalnum)
    except:
        return None
    return a,label