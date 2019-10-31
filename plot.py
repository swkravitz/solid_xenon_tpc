import numpy as np
import pylab as pl
import matplotlib as mpl

# set plotting style
mpl.rcParams['font.size']=14
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[16.0,12.0]

channel_0=np.fromfile("A-50.01V-extT.dat", dtype="int16")
channel_1=np.fromfile("A-50.01V-extT.dat", dtype="int16")
channel_2=np.fromfile("A-50.01V-extT.dat", dtype="int16")
channel_3=np.fromfile("A-50.01V-extT.dat", dtype="int16")
vscale=(2000.0/16384.0)
wsize=50000
V=vscale*channel_0
V_1=vscale*channel_1
V_2=vscale*channel_2
V_3=vscale*channel_3
n_channels=4
v_matrix = V.reshape((V.size//wsize),wsize)
v1_matrix = V_1.reshape((V.size//wsize),wsize)
v2_matrix = V_2.reshape((V.size//wsize),wsize)
v3_matrix = V_3.reshape((V.size//wsize),wsize)
v4_matrix = v_matrix+v1_matrix+v2_matrix+v3_matrix
v_matrix_all_ch=[v_matrix,v1_matrix,v2_matrix,v3_matrix,v4_matrix]
x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0)
t=tscale*x
t_matrix=np.repeat(t[np.newaxis,:], V.size/wsize, 0)
max_ind_array=np.zeros((v_matrix.shape[0],4) )
max_val_array=np.zeros((v_matrix.shape[0],4) )
integral_array=np.zeros((v_matrix.shape[0],4) )

print(v_matrix.shape[0])
for i in range (0, v_matrix.shape[0]):
    # for each channel
    for j in range(0, n_channels):
        #i = input("Window number between 1 and " + str((V.size/wsize)) + ": ")
        
        baseline=np.mean(v_matrix_all_ch[j][i,:500]) #avg ~1 us
        #print("baseline: ",baseline)
        
        win_min=int(18./tscale)
        win_max=int(21./tscale)
        integral=np.sum(v_matrix_all_ch[j][i,win_min:win_max]-baseline)
        integral_array[i,j]=integral
        # Threshold integral for trigger at 770: 1.9e5
        
        #print("Below is max index")
        max_ind=tscale*np.argmax(v_matrix_all_ch[j][i,:])
        max_ind_array[i,j]=max_ind
        #print(max_ind) 
        #print("Below is max value")
        max_val=np.max(v_matrix_all_ch[j][i,:])
        max_val_array[i,j]=max_val
        #print(max_val)
        
    # once
    if -9000000000000<integral:
        pl.figure(1,figsize=(20, 20))
        pl.clf()
        pl.rc('xtick', labelsize=14)
        pl.rc('ytick', labelsize=14)
       
        pl.subplot(2,2,1)
        pl.plot(t_matrix[i,:],v_matrix[i,:],'y',linewidth=4.5)
        pl.xlim([18.5, 22.5])
        pl.ylim([0, 2000])
        pl.xlabel('Time [us]')
        pl.ylabel('Millivolts')
        pl.title("SiPM A")
        triggertime_us = (t[-1]*0.2)
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,2,2)
        pl.plot(t_matrix[i,:],v1_matrix[i,:],'cyan', linewidth=4.5)
        pl.xlim([18.5, 22.5])
        pl.ylim([0, 2000])
        pl.xlabel('Time [us]')
        pl.ylabel('Millivolts')
        pl.title("SiPM B")
        triggertime_us = (t[-1]*0.2)
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,2,3)
        pl.plot(t_matrix[i,:],v2_matrix[i,:],'magenta', linewidth=4.5)
        pl.xlim([18.5, 22.5])
        pl.ylim([0, 2000])
        pl.xlabel('Time [us]')
        pl.ylabel('Millivolts')
        pl.title("SiPM C")
        triggertime_us = (t[-1]*0.2)
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,2,4)
        pl.plot(t_matrix[i,:],v3_matrix[i,:],'blue', linewidth=4.5)
        pl.xlim([18.5, 22.5])
        pl.ylim([0, 2000])
        pl.xlabel('Time [us]')
        pl.ylabel('Millivolts')
        pl.title("SiPM D")
        triggertime_us = (t[-1]*0.2)
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.show(0)
        inn = input("Press enter to continue")

pl.figure(figsize=(20, 20))
pl.clf()     
for j in range(0, n_channels):   
    pl.subplot(2,2,j+1)
    pl.hist(max_ind_array[:,j],bins=100)
    pl.yscale('log')
    pl.xlabel("Time of max value")
pl.figure(figsize=(20, 20))
pl.clf()
for j in range(0, n_channels):    
    pl.subplot(2,2,j+1)  
    pl.hist(max_val_array[:,j],bins=100)
    pl.xlabel("Max value")
    pl.yscale('log')
pl.figure(figsize=(20, 20))
pl.clf()
for j in range(0, n_channels):    
    pl.subplot(2,2,j+1)
    pl.hist(integral_array[:,j],bins=100,range=(-100,1500))
    pl.xlabel("Pulse integral")
    #pl.yscale('log')
    pl.title('Ch '+str(j))
pl.show(0)
