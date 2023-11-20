import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from DataCollection import *
from scipy.fft import *
from mpl_toolkits.mplot3d import Axes3D

def Generate_FTGraph(data_origin, bin_size, HFL):
    data_length = np.size(data_origin)
    amp = np.zeros((int(np.ceil(data_length/bin_size)),(int(np.floor(bin_size/2)-1))))
    phase = np.zeros((int(np.ceil(data_length/bin_size)),(int(np.floor(bin_size/2)-1))))
    time_bin = 0
    for start in range(0,data_length,bin_size):
        cel = np.minimum(start+bin_size,data_length)
        fa = fft(data_origin[start:cel])/(cel-start)
        half_fa = fa[:(int(np.floor((cel-start)/2)-1))]
        amp[time_bin,:(int(np.floor((cel-start)/2)-1))] = np.abs(half_fa)
        phase[time_bin,:(int(np.floor((cel-start)/2)-1))] = np.angle(half_fa)
        time_bin += 1
    return amp[:,:HFL],phase[:,:HFL]

def Generate_FFT(data_origin):
    data_length = np.size(data_origin)
    fa = fft(data_origin)/data_length
    half_fa = fa[:(int(np.floor(data_length/2)-1))]
    amp = np.abs(half_fa)
    phase = np.angle(half_fa)
    return amp,phase

# if __name__ == "__main__":
#     dir = cmod_dir+r"\CMod_train\1120000000\\"
#     for filename in os.listdir(dir):
#         shotno_str = filename.split(".")[0]
#         print("Shotno: "+shotno_str)
#         f = Get_One_Shot(dir+filename)
#         time = Get_Time(f)
#         print("Start/Down Time: ", time)
#         IsDisrupt = Get_IsDisrupt(f)
#         print("Disrupt: ",IsDisrupt)

#         a = Get_Data_Entry(f,'\\MAGNETICS::TOP.ACTIVE_MHD.SIGNALS:BP01_GHK')
#         plt.plot(a)
#         plt.show()
#         amp,phase = Generate_FTGraph(a,200)

#         shape_amp = np.shape(amp)
#         x = np.arange(shape_amp[0])
#         y = np.arange(shape_amp[1])
#         X,Y = np.meshgrid(x,y)

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(X.transpose(),Y.transpose(),amp)
#         plt.show()
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot_surface(X.transpose(),Y.transpose(),phase)
#         plt.show()
