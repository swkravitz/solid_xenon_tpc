import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

biaslist = np.array([54,55,56,57])
#mode = "randomDC" 
mode = "led"
tscale = (8.0/4096.0)

#data_dir = "../../led4.04_old_rq/"
#data_dir = "../../darkcount_cold/"
data_dir = "/Volumes/Samsung USB/cold_led4.04_b%dv/"
#data_dir = "/Volumes/Samsung USB/cold_dark_b%dv/"
filebase = "spe_rq_"
fontsize = 14
pl.rc('xtick',labelsize=fontsize)
pl.rc('ytick',labelsize=fontsize)
def gauss(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

#for channel in range(16):
for channel in [12]:
    gainlist = []
    gainerrlist = []
    for bias in biaslist:
         rq = np.load(data_dir%bias+filebase+"b%dV.npz"%bias)
#         rq = np.load(data_dir%bias+filebase+"%d.npz"%bias)
         area = rq["p_area_%d"%channel]*tscale 
         area = area[area!=0]
         nbins = 80
         uplim = 0.07
         lowlim = -0.01
         [data,bins] = np.histogram(area, bins=nbins, range=(lowlim,uplim))
         binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])
         if mode == "randomDC":
             noisearea = rq["p_noisearea_%d"%channel]*tscale
             #set the initial fit range for the sphe peak
             fmin = 0
             fmax = 0.02
             #preliminary fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
             x = binCenters[gpts]
             y = data[gpts]
             mean = x[np.argmax(y)]
             try:
                 [p0,p0cov] = curve_fit(gauss, xdata=x, ydata=y, p0=[150,mean,0.002])
             except:
                 print("can't perform intial sphe fit on channel %d"%channel)
                 pl.figure()
                 pl.hist(area,nbins,range=(lowlim,uplim))
                 pl.show()
                 break
             #set the fit range based on the intial fit
             fmin = p0[1]-p0[2]             
             fmax = p0[1]+p0[2]             
             #final fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
             x = binCenters[gpts]
             y = data[gpts]
             [p,pcov] = curve_fit(gauss, xdata=x, ydata=y, p0=p0)

             fmin = p0[1]-1.5*p0[2]             
             fmax = p0[1]+1.5*p0[2]             
             #final fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
             x3 = binCenters[gpts]
             y3 = data[gpts]
             [p3,p3cov] = curve_fit(gauss, xdata=x3, ydata=y3, p0=p0)

             perr = np.sqrt(np.diag(pcov))
             gainlist.append(p[1])
             gainerrlist.append(perr[1])
             #plot pulse area hitos and fitted guassians
             pl.figure()
             areaandbase = np.concatenate([area,noisearea[0:5000]])
#             pl.hist(area,nbins,range=(lowlim,uplim))
             pl.hist(areaandbase,nbins,range=(lowlim,uplim))
             pl.plot(x,gauss(x, *p), color='blue',label="mean=%.4f"%p[1])
             pl.plot(x3,gauss(x3, *p3), color='red',label="mean=%.4f"%p3[1])
#             pl.xlim([0,uplim])
             xticksteps = 0.01
             pl.text(uplim/4.,pl.ylim()[1]*0.6,"Bias = %dV\nMean = %.4f mV*us\nSigma = %.4f mV*us"%(bias,p[1],p[2]))
             pl.xticks(np.arange(lowlim,uplim+xticksteps, xticksteps))
             pl.title("Channel %d"%channel,fontsize=fontsize)
             pl.xlabel("Pulse Area (mV*us)", fontsize=fontsize)
             pl.legend(loc="upper right",prop={"size":fontsize})
    #         pl.savefig("Channel_%d_Bias_%dV"%(channel,bias))
             pl.show()

         elif mode == "led":
             # fit the noise peak 
             noisemean = binCenters[np.argmax(data)]
#             gpts = np.logical_and(0.<binCenters, noisemean*2.2>binCenters)
             gpts = 0.01>binCenters
             xnoise = binCenters[gpts]
             ynoise = data[gpts]
             try:
                 [noisep,noisepcov] = curve_fit(gauss, xdata=xnoise, ydata=ynoise, p0=[16000,noisemean,0.01],bounds=[[100,-0.1,0],[50000,0.1,0.2]])
             except:
                 print("can't fit noise peak on channel %d"%channel)
                 pl.figure()
                 pl.hist(area,nbins,range=(lowlim,uplim))
                 pl.show()
                 continue
             #set the initial fit range for the sphe peak
             fmin = noisep[1]+3*noisep[2] #left bound is 4 sigma away from the noise peak
             fmax = fmin+0.015
             #preliminary fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
             x = binCenters[gpts]
             y = data[gpts]
             mean = x[np.argmax(y)]
             try:
                 [p0,p0cov] = curve_fit(gauss, xdata=x, ydata=y, p0=[150,mean,0.002])
             except:
                 print("can't perform intial sphe fit on channel %d"%channel)
                 pl.figure()
                 pl.hist(area,nbins,range=(lowlim,uplim))
                 pl.plot(xnoise,gauss(xnoise, *noisep))
                 pl.plot([fmin,fmax],[100,100])
                 pl.plot(mean,y[np.argmax(y)],"o")
                 pl.show()
                 break
             #set the fit range based on the intial fit
             fmax = p0[1]+2*p0[2]
             #final fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
             x = binCenters[gpts]
             y = data[gpts]
             [p,pcov] = curve_fit(gauss, xdata=x, ydata=y, p0=p0)
             perr = np.sqrt(np.diag(pcov))
             gainlist.append(p[1])
             gainerrlist.append(perr[1])
             #plot pulse area hitos and fitted guassians
             pl.figure()
             pl.hist(area,nbins,range=(lowlim,uplim))
             pl.plot(xnoise,gauss(xnoise, *noisep), color='orange')
             pl.plot(x,gauss(x, *p), color='red')
             pl.xlim([lowlim,uplim])
             xticksteps = 0.01
             pl.text(uplim/4.,pl.ylim()[1]*0.6,"Bias = %dV\nMean = %.4f mV*us\nSigma = %.4f mV*us"%(bias,p[1],p[2]))
             pl.xticks(np.arange(lowlim,uplim+xticksteps, xticksteps))
             pl.title("Channel %d"%channel,fontsize=fontsize)
             pl.xlabel("Pulse Area (mV*us)",fontsize=fontsize)
    #         pl.savefig("Channel_%d_Bias_%dV"%(channel,bias))
             pl.show()

    #plot gain curve for the given channel
    if len(gainlist)>1:
        gainlist = np.array(gainlist)
        gainconv = pow(10,-3)*pow(10,-6)/(1.602*pow(10,-19)*25)
        gainlist = gainlist*gainconv
        m,b = np.polyfit(biaslist,gainlist,1)
        print (biaslist,gainlist)
        pl.figure()
        pl.xlabel("Bias voltage (V)",fontsize=fontsize)
        pl.ylabel("SiPM Gain",fontsize=fontsize)
        if b>0:
            sign = "+"
        else:
            sign = "-"
        pl.plot(biaslist,m*biaslist+b,"b",label="%s:%d*x%s%d"%("dark count",m,sign,abs(b)))
        pl.errorbar(biaslist,gainlist,yerr=gainerrlist,fmt='o',color="b")
#        gainlist2_old = [2045810.67910546,3153525.80880244,3774032.33859399,4291897.25896552]
        gainlist2 = [2656858.00862382,3301362.05565534,3842214.58339314,4290514.78103186]
        m2,b2 = np.polyfit(biaslist[2:],gainlist2[2:],1)
        pl.plot(biaslist,m2*biaslist+b2,"g",label="%s:%d*x%s%d"%("led",m2,sign,abs(b2)))
        pl.plot(biaslist,gainlist2,"go")
        pl.legend(loc="upper left",prop={"size":fontsize})
        pl.title("Channel %d"%channel,fontsize=fontsize)
#        pl.savefig(data_dir+"/gaincurve_ch%d"%channel)
        pl.show()

