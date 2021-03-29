import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

biaslist = np.array([54,55,56,57])

tscale = (8.0/4096.0)/4.0

data_dir = "../../led4.04_old_rq/"
filebase = "spe_rq_"

def gauss(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

for channel in range(16):
    gainlist = []
    gainerrlist = []
    for bias in biaslist:
         rq = np.load(data_dir+filebase+"%d.npz"%bias)
         area = rq["p_area_%d"%channel]*tscale 
         nbins = 150
         uplim = 0.07
         [data,bins] = np.histogram(area, bins=nbins, range=(0,uplim))
         binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])
         # fit the noise peak 
         noisemean = binCenters[np.argmax(data)]
         gpts = np.logical_and(0.<binCenters, noisemean*2.2>binCenters)
         xnoise = binCenters[gpts]
         ynoise = data[gpts]
         try:
             [noisep,noisepcov] = curve_fit(gauss, xdata=xnoise, ydata=ynoise, p0=[4000,noisemean,0.002],bounds=[[100,0.,0],[5000,0.1,0.2]])
         except:
             print("can't fit noise peak on channel %d"%channel)
             pl.figure()
             pl.hist(area,nbins,range=(0,uplim))
             pl.show()
             break
         #set the initial fit range for the sphe peak
         fmin = noisep[1]+4*noisep[2] #left bound is 4 sigma away from the noise peak
         fmax = fmin+0.015
         print (fmin,fmax)
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
             pl.hist(area,nbins,range=(0,uplim))
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
         pl.hist(area,nbins,range=(0,uplim))
         pl.plot(xnoise,gauss(xnoise, *noisep), color='orange')
         pl.plot(x,gauss(x, *p), color='red')
         pl.xlim([0,uplim])
         xticksteps = 0.01
         pl.text(uplim/4.,1000,"Bias = %dV\nMean = %.4f mV*us\nSigma = %.4f mV*us"%(bias,p[1],p[2]))
         pl.xticks(np.arange(0,uplim+xticksteps, xticksteps))
         pl.title("Channel %d"%channel)
         pl.xlabel("Pulse Area (mV*us)")
#         pl.savefig("Channel_%d_Bias_%dV"%(channel,bias))
#         pl.show()

    #plot gain curve for the given channel
    if len(gainlist)>1:
        gainlist = np.array(gainlist)
        gainconv = pow(10,-3)*pow(10,-6)/(1.602*pow(10,-19)*25)
        gainlist = gainlist*gainconv
        m,b = np.polyfit(biaslist,gainlist,1)
        pl.figure()
        pl.xlabel("Bias voltage (V)")
        pl.ylabel("SiPM Gain")
        if b>0:
            sign = "+"
        else:
            sign = "-"
        pl.plot(biaslist,m*biaslist+b,"b",label="%d*x%s%d"%(m,sign,abs(b)))
        pl.errorbar(biaslist,gainlist,yerr=gainerrlist,fmt='o',color="b")
        pl.legend(loc="upper right")
        pl.title("Channel %d"%channel)
        pl.savefig("gaincurve_ch%d"%channel)
        pl.show()

