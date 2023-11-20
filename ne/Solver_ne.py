import numpy as np
import torch
import torch.nn as nn
from LuNet7_ne import *
from DataCollection import *
from MagneticAnalysis import *

def Train_and_Validate(net,optimizer,scheduler,criterion,max_norm, train_shotlist,validate_shotlist,max_epoch,break_loss):
    train_setsize = len(train_shotlist)
    validate_setsize = len(validate_shotlist)

    Training_Loss = []
    Validate_Loss = []
    Train_accu = []
    Validate_accu = []

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(max_epoch):
        Train_Loss_Epoch,T_accu = Train_NE(net,optimizer,scheduler,criterion,train_shotlist,epoch)
        Validate_Loss_Epoch,V_accu = Validate_NE(net,criterion,validate_shotlist,epoch)
        Training_Loss += Train_Loss_Epoch
        Validate_Loss += Validate_Loss_Epoch
        Train_accu.append(T_accu)
        Validate_accu.append(V_accu)
        if np.max([np.max(Train_Loss_Epoch),np.max(Validate_Loss_Epoch)])<break_loss:
            break
    return Training_Loss, Validate_Loss,Train_accu,Validate_accu

def Train_NE(net,optimizer,scheduler,criterion,train_shotlist,epoch):
    Train_Loss_Epoch = []
    input_data, label= Sample_Generator(train_shotlist)
    seq_num = len(input_data)
    correct = 0
    for seq in range(seq_num):
        output = net(input_data[seq])
        # print(output.detach(),label[seq].detach())
        loss = criterion(output,label[seq])
        loss.backward()
        nn.utils.clip_grad_norm_(net.fc.parameters(),0.5)
        optimizer.step()
        average_loss = loss.sum().item()
        # for i in range(input_data[seq].shape[0]):
        # print(label[seq].detach().to('cpu'),output.detach().to('cpu'))
        # plt.plot((input_data[seq,0].reshape(-1).detach().to('cpu')))
        # # plt.legend()
        # plt.show()
        prediction = torch.round(output.detach())
        gt = label[seq].detach()
        correct += torch.eq(prediction,gt).sum().item()
        # print(prediction,gt)
        # total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # print(f'Total number of trainable parameters: {total_trainable_params}')
        # for name, param in net.named_parameters():
        #     if param.grad is not None:
        #         print(f'Parameter: {name}, Avg Rel Gradient: {param.grad.mean()/param.mean()}, Avg Gradient: {param.grad.mean()}')
        
        Train_Loss_Epoch.append(average_loss)
        print('Train: Epoch [%u]\t [%u]/[%u] Loss[%.04f]'%(epoch, seq, seq_num, average_loss))
    print(correct/seq_num)
    scheduler.step()
    return Train_Loss_Epoch,correct/seq_num

def Validate_NE(net,criterion,validate_shotlist,epoch):
    Validate_Loss_Epoch = []
    input_data, label = Sample_Generator(validate_shotlist)
    # print(label)
    seq_num = len(input_data)
    correct = 0
    for seq in range(seq_num):
        output = net(input_data[seq])
        loss = criterion(output,label[seq])
        average_loss = loss.sum().item()
        print(label[seq].detach().to('cpu'),output.detach().to('cpu'))
        prediction = torch.round(output.detach())
        gt = label[seq].detach()
        correct += torch.eq(prediction,gt).sum().item()
        # accu = correct/batch_size
        Validate_Loss_Epoch.append(average_loss)
        print('Validate: Epoch [%u]\t [%u]/[%u] Loss[%.04f]'%(epoch, seq, seq_num, average_loss))
    print(correct/seq_num)
    return Validate_Loss_Epoch,correct/seq_num

def Sample_Generator(filelist):
    signalnum = 5
    Norm_Length = 2000
    data = []
    label = []
    for file in filelist:
        try:
            a,isdisrupt,_ = Get_signal_from_shot(file,signalnum)
        except:
            continue

        a = np.array(a[:-60])
        if np.size(a) <0.75*Norm_Length:
            a = np.append(a,np.zeros(Norm_Length-np.size(a)))
    # a = (np.array(a)-67)*0.3 #C-Mod Dx
        a = np.abs(a/1e20)-np.mean(np.abs(a/1e20))

        
        a = Size_Normalization(a,Norm_Length)

        data.append(a)
        label.append(isdisrupt)

    data = torch.tensor(np.array(data)).reshape(-1,1,Norm_Length).to(torch.float32).to('cuda')
    label = torch.tensor(np.array(label).reshape(-1,1)).to(torch.float32).to('cuda')
    return data,label