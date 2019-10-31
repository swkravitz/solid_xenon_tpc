import numpy as np
import pylab as pl
import matplotlib as mpl

# set plotting style
mpl.rcParams['font.size']=14
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[16.0,12.0]

channel_0=np.fromfile("A-50.01V-extT.dat", dtype="int16")
vscale=(2000.0/16384.0)
wsize=50000
V=vscale*channel_0
v_matrix = V.reshape((V.size//wsize),wsize)

x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0)
t=tscale*x
#t_matrix=np.repeat(t[np.newaxis,:], V.size/wsize, 0)
#max_ind_array=np.zeros((v_matrix.shape[0],4) )
#max_val_array=np.zeros((v_matrix.shape[0],4) )
integral_array=np.zeros(v_matrix.shape[0])

print(v_matrix.shape[0])

for i in range(0, v_matrix.shape[0]):
    baseline=np.mean(v_matrix[:500])

    win_min=int(18./tscale)
    win_max=int(21./tscale)
    integral=np.sum(v_matrix[i,win_min:win_max]-baseline)
    integral_array[i]=integral

pl.hist(integral_array,bins=100,range=(-100,1500))
pl.xlabel("ADC [counts]")
#pl.yscale('log')
pl.title('Pulse Integral Spectrum')
pl.show(0)
