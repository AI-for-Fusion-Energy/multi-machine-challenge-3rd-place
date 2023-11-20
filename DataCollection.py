import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
from scipy.interpolate import interp1d
from MagneticAnalysis import *


hl2a_dir = r".\HL-2A_data\HL-2A data\HL-2A_Data\JDDB_repo_2A_5k"
jtext_dir = r".\J-TEXT_data\J-TEXT data\data\processed_data_1k_5k_final"
cmod_dir = r".\C-Mod_data\New C-mod data"

def Get_One_Shot(filename):
    f = h5py.File(filename,'r')
    return f

def Get_Data_Entry(f,signalname):
    a = np.array(f['data'][signalname])
    return a

def Get_IsDisrupt(f):
    a = np.array(f['meta']['IsDisrupt'])
    return a

def Get_Time(f):
    a = np.array(f['meta']['StartTime'])
    b = np.array(f['meta']['DownTime'])
    return np.append(a,b)
    
def Overview(dir):
    for filename in os.listdir(dir):
        shotno_str = filename.split(".")[0]
        print("Shotno: "+shotno_str)
        f = Get_One_Shot(dir+filename)
        time = Get_Time(f)
        print("Start/Down Time: ", time)
        IsDisrupt = Get_IsDisrupt(f)
        print("Disrupt: ",IsDisrupt)
        a = Get_Data_Entry(f,'\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP01_GHK')
        # time_axis = np.arange(time[0],time[1],(time[1]-time[0])/np.size(a))
        # plt.plot(time_axis,a)
        plt.plot(a)
        plt.title(shotno_str+'\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP01_GHK')
        plt.show()

def Dataname_Searchtable(jch,row):
    with open('ITU data - data.csv','r') as file:
        csv_reader = csv.reader(file)
        i = 0
        for controw in csv_reader:
            if i != row:
                i += 1
                continue
            if jch == 'J':
                return controw[3]
            elif jch == 'C':
                if '&' in controw[4]:
                    controw4 = controw[4].split(' & ')
                    return controw4[0]
                else:
                    return controw[4]
            elif jch == 'H':
                return controw[5]
            else:
                raise Exception('No such device')

def Check_Which_Device(filename):
    contline = filename.split('\\')
    cont1 = contline[1]
    if 'J-TEXT' in cont1:
        return "J"
    elif 'C-Mod' in cont1:
        return "C"
    elif 'HL-2A' in cont1:
        return "H"
    else:
        raise Exception('No such device')

def Generate_Batch():
    jtext_shotlist = []
    for dirname in os.listdir(jtext_dir):
        for filename in os.listdir(jtext_dir+'\\'+dirname):
            jtext_shotlist.append(jtext_dir+'\\'+dirname+'\\'+filename)
    hl2a_shotlist = []
    for filename in os.listdir(hl2a_dir):
        hl2a_shotlist.append(hl2a_dir+'\\'+filename)
    cmod_train_shotlist = []
    for filename in os.listdir(cmod_dir+'\\'+r'CMod_train\1120000000'):
        cmod_train_shotlist.append(cmod_dir+'\\'+r'CMod_train\1120000000'+'\\'+filename)

    total_shotlist = cmod_train_shotlist
    random.shuffle(total_shotlist)
    train_shotlist = total_shotlist[:16]
    validate_shotlist = total_shotlist[16:]

    # train_shotlist = jtext_shotlist+hl2a_shotlist
    # validate_shotlist = cmod_train_shotlist

    # train_shotlist = cmod_train_shotlist[:16]
    # validate_shotlist = cmod_train_shotlist[16:]
    random.shuffle(train_shotlist)
    random.shuffle(validate_shotlist)

    return train_shotlist,validate_shotlist

def Get_signal_from_shotlist(filelist,signalnum):
    a = []
    isdisrupt = []
    time = []
    for file in filelist:
        jch = Check_Which_Device(file)
        signalname = Dataname_Searchtable(jch,signalnum)
        f = Get_One_Shot(file)
        a.append(Get_Data_Entry(f, signalname))
        isdisrupt.append(Get_IsDisrupt(f))
        time.append(Get_Time(f))
        f.close()
    return a, isdisrupt,time

def Get_signal_from_shot(file,signalnum):
    jch = Check_Which_Device(file)
    signalname = Dataname_Searchtable(jch,signalnum)
    f = Get_One_Shot(file)
    a=Get_Data_Entry(f, signalname)
    isdisrupt=Get_IsDisrupt(f)
    time=Get_Time(f)
    f.close()
    return a, isdisrupt,time

def Size_Normalization(a,n = 2000):
    size_origin = len(a)
    x = np.arange(size_origin)/(size_origin-1)
    F = interp1d(x,a)
    newx = np.arange(n)/n
    proceeded=F(newx)
    return proceeded

def Prepare_Magnetic_Data(a_list):
    amp_list = []
    phase_list = []
    for a in a_list:
        amp, phase = Generate_FTGraph(a, 200)
        amp_list.append(amp)
        phase_list.append(phase)
    return amp_list, phase_list


def Generate_Test_shotlist():
    test_shotlist = []
    for dirname in os.listdir(cmod_dir+'\\'+r'CMod_evaluate'):
        if dirname == '.DS_Store':
            continue
        for filename in os.listdir(cmod_dir+'\\'+r'CMod_evaluate'+'\\'+dirname):
            test_shotlist.append(cmod_dir+'\\'+r'CMod_evaluate'+'\\'+dirname+'\\'+filename)
    return test_shotlist

def Get_Test_Signal(file,signal_num):
    jch = Check_Which_Device(file)
    signalname = Dataname_Searchtable(jch,signal_num)
    f = Get_One_Shot(file)
    a = Get_Data_Entry(f, signalname)
    f.close()
    return a

def Get_Shotno(file):
    cont = file.split('\\')
    shotno = cont[-1].split('.')[0]
    return shotno

def Get_Time_until_disrupt(f):
    a = np.array(f['data']['time_until_disrupt'])
    return a

def Chop_End(a_list,time,chop,warning_ratio = 0.15):
    proceeded = []
    for i in range(len(a_list)):
        a = a_list[i]
        chop_bins = int(chop/(np.max(time)-np.min(time))*len(a))
        if chop/(np.max(time)-np.min(time))>=warning_ratio:
            chop_bins = int(warning_ratio*len(a))
        elif chop_bins<=1:
            chop_bins = 1
        proceeded.append(a[:(-chop_bins)])
    return proceeded

def Signal_Check():
    jtext_shotlist = []
    for dirname in os.listdir(jtext_dir):
        for filename in os.listdir(jtext_dir+'\\'+dirname):
            jtext_shotlist.append(jtext_dir+'\\'+dirname+'\\'+filename)
    hl2a_shotlist = []
    for filename in os.listdir(hl2a_dir):
        hl2a_shotlist.append(hl2a_dir+'\\'+filename)
    cmod_train_shotlist = []
    for filename in os.listdir(cmod_dir+'\\'+r'CMod_train\1120000000'):
        cmod_train_shotlist.append(cmod_dir+'\\'+r'CMod_train\1120000000'+'\\'+filename)
    
    test_shotlist = cmod_train_shotlist
    # test_shotlist = jtext_shotlist[:20]+hl2a_shotlist[:20]+cmod_train_shotlist
    # test_shotlist = Generate_Test_shotlist()
    # test_shotlist = hl2a_shotlist
    random.shuffle(test_shotlist)
    for file in test_shotlist:
        jch = Check_Which_Device(file)
        signalname = Dataname_Searchtable(jch,91)
        f = Get_One_Shot(file)
        a = Get_Data_Entry(f, signalname)
        isdisrupt = Get_IsDisrupt(f)
        time = Get_Time(f)
        tud = Get_Time_until_disrupt(f)
        tud_min = np.min(tud)
        print(jch,signalname,isdisrupt,time, len(a),tud_min)
        plt.plot(a)
        plt.title(jch)
        plt.show()

# signal_list = [1,3,4,5,8,9,12,32,72,89,90,91,92,93]


if __name__ == "__main__":
    # Overview(cmod_dir+r"\CMod_train\1120000000\\")
    Signal_Check()