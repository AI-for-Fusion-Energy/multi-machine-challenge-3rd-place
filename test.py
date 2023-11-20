import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from DataCollection import *
import random
import csv
import torch
from dx.LuNet7_dx import LuNet7_dx
from q95.LuNet7_q95 import LuNet7_q95
from n1n.LuNet7_n1norm import LuNet7_n1norm
from dy.LuNet7_dy import LuNet7_dy
from ip.LuNet7_ip import LuNet7_ip
from ipe.LuNet7_ipe import LuNet7_ipe
from vl.LuNet7_vl import LuNet7_vl
from ne.LuNet7_ne import LuNet7_ne
from c3.LuNet7_c3 import LuNet7_c3
from vote.LuNet_vote import LuNet_vote

def Test_Sample_Generator(file,signalnum):
    try:
        a,_,_ = Get_signal_from_shot(file,signalnum)
    except:
        return None
    return a

def Test_Sample_Normalize(a,signalnum):
    length = []
    if(signalnum==91):
        Norm_Length = 2000
        a = np.nan_to_num(a)
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/0.1)-np.mean(np.abs(a/0.1))
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum==90):
        Norm_Length = 2000
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/5)-np.mean(np.abs(a/5))
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum==89):
        Norm_Length = 2000
        a = np.nan_to_num(a)
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/1e-3)-np.mean(np.abs(a/1e-3))
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum==9):
        Norm_Length = 2000
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        if a[0] < 0:
            a = -a
        a = a-np.mean(a)
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum==8):
        Norm_Length = 20000
        istcs = IsTCS(a,Norm_Length,alpha = 0.78)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/300)-np.mean(np.abs(a/300))
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum==5):
        Norm_Length = 2000
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/1e20)-np.mean(np.abs(a/1e20))
        a = Size_Normalization(a,Norm_Length)
        
    if(signalnum in [3,4]):
        Norm_Length = 2000
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = Size_Normalization(a,Norm_Length)

    if(signalnum==1):
        Norm_Length = 2000
        istcs = IsTCS(a,Norm_Length)
        if istcs:
            return istcs
        length.append(np.size(a))
        a = np.abs(a/1e6)-np.mean(np.abs(a/1e6))
        a = Size_Normalization(a,Norm_Length)
    # print(length)
    data = torch.tensor(np.array(a)).reshape(1,1,-1).to(torch.float32).to('cuda')
    return data

def IsTCS(a, Norm_Length, alpha = 0.85):
    if len(a) < alpha*Norm_Length:
        return True
    else:
        return False

def Net_init(signalnum):
    if(signalnum==1):
        net = LuNet7_ip().to('cuda')
        net.load_state_dict(torch.load('./ip/LuNet7_ip.pkl'))
        return net
    if(signalnum==3):
        net = LuNet7_dx().to('cuda')
        net.load_state_dict(torch.load('./dx/LuNet7_dx.pkl'))
        return net
    if(signalnum==4):
        net = LuNet7_dy().to('cuda')
        net.load_state_dict(torch.load('./dy/LuNet7_dy.pkl'))
        return net
    if(signalnum==5):
        net = LuNet7_ne().to('cuda')
        net.load_state_dict(torch.load('./ne/LuNet7_ne.pkl'))
        return net
    if(signalnum==8):
        net = LuNet7_c3().to('cuda')
        net.load_state_dict(torch.load('./c3/LuNet7_c3.pkl'))
        return net
    if(signalnum==9):
        net = LuNet7_vl().to('cuda')
        net.load_state_dict(torch.load('./vl/LuNet7_vl.pkl'))
        return net
    if(signalnum==89):
        net = LuNet7_n1norm().to('cuda')
        net.load_state_dict(torch.load('./n1n/LuNet7_n1norm.pkl'))
        return net
    if(signalnum==90):
        net = LuNet7_q95().to('cuda')
        net.load_state_dict(torch.load('./q95/LuNet7_q95.pkl'))
        return net
    if(signalnum==91):
        net = LuNet7_ipe().to('cuda')
        net.load_state_dict(torch.load('./ipe/LuNet7_ipe.pkl'))
        return net

if __name__ == '__main__':
    test_shotlist = Generate_Test_shotlist()
    data = [['Shot_list','Is_disrupt']]
    disrupt_count = 0
    batch_size = 1
    signalnum_list = [1,3,4,5,8,9,89,90,91]
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    net_vote = LuNet_vote()
    net_vote.load_state_dict(torch.load('./vote/LuNet_vote.pkl'))


    for file in test_shotlist:
        output = []
        for signalnum in signalnum_list:
            a = Test_Sample_Generator(file,signalnum)
            if a is None:
                output.append(-1)
                continue
            
            input_data = Test_Sample_Normalize(a,signalnum)
            if isinstance(input_data,bool):
                output.append(1)
                continue
            net = Net_init(signalnum)
            output.append(net(input_data[0]).reshape(1).detach().item())
        output = torch.tensor(np.array(output)).to(torch.float32)
        if(output.max().detach().item() ==1):
            prediction = 1
        else:
            vote = net_vote(output).detach().item()
            print(vote)
            if(vote==1):
                prediction = 1
            else:
                prediction = 0
        
        shotno = Get_Shotno(file)
        data.append(['ID_'+shotno, prediction])
        if(prediction==1):
            disrupt_count += 1
        # for i in range(datain.shape[0]):
        #     shape_amp = np.shape(datain[i,0].detach())
        #     a = np.arange(shape_amp[0])
        #     b = np.arange(shape_amp[1])
        #     A,B = np.meshgrid(a,b)

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot_surface(A.transpose(),B.transpose(),datain[i,0].detach())
        #     plt.show()
        #     print(prediction)
    print('total: %u, disrupt: %u'%(len(test_shotlist),disrupt_count))
    filepath = './submit.csv'
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


