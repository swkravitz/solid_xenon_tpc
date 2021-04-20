import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#templist = np.array([117.6,110.0,103.1,97.3,90.2])
#templist = np.array([117.6,110.0,103.1])
templist = np.array([100.6])

tscale = (8.0/4096.0)*1000 #ns

#data_dir = "/Volumes/Samsung USB/crystalize/dark_count_all_channel_Cu_rod_%s/"
data_dir = "/Users/qingxia/Documents/Physics/LZ/SiPM/dark_count_all_ch_calibration_%s/"
filebase = "spe_rq_"
fontsize = 14
impedance = 50 #input impedance
gainconv = pow(10,-3)*pow(10,-9)/(1.602*pow(10,-19)*impedance)
pl.rc('xtick',labelsize=fontsize)
pl.rc('ytick',labelsize=fontsize)
def gauss(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))
def linearfit(x,a,b): 
     return a*x+b
fig = pl.figure()
ax = fig.gca()
for channel in range(8):
#for channel in [7]:
    gainlist = []
    gainerrlist = []
    plottemplist = []
    for temp in templist:
         if channel==0:
             outfile = open("SiPMgain_-%sC.txt"%str(temp),"w")
             outfile.write("Channel;Gain;GainError\n")
         else:
             outfile = open("SiPMgain_-%sC.txt"%str(temp),"a")
         peaklist = []
         peakerrlist = []
         if channel==7 and temp ==100.6:
             rq = np.load(data_dir%str(temp)+"randomDC_"+filebase+"t-%sC.npz"%str(temp))
             area = rq["p_area_%d"%channel]*tscale 
             noisearea = rq["p_noisearea_%d"%channel]*tscale
             area = np.concatenate([area,noisearea[0:5000]])
         else:
             rq = np.load(data_dir%str(temp)+filebase+"t-%sC.npz"%str(temp))
             area = rq["p_area_%d"%channel]*tscale 
         area = area[area!=0]
#         nbins = 180
         nbins = 100
         uplim = 130
         lowlim = -20
         [data,bins] = np.histogram(area, bins=nbins, range=(lowlim,uplim))
         binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])

         #find prominent peaks in the dark count spectrum
         min_height = 300
         min_width = 1
         rel_height = 0.5 
         prominence = 50
         peaks, properties = find_peaks(data, height=min_height, width=min_width, rel_height=rel_height, prominence=prominence)
         peakmeans = binCenters[peaks]
         #set the initial gaussain fit range for the peaks
         subfig = pl.figure()
         subax = subfig.gca()
         subax.hist(area,nbins,range=(lowlim,uplim))
         for i in range(len(peakmeans)):
             lab=""
             lab2 = "final fit"
             try:
                 if len(peakmeans)==1:
                     print ("only one peak found on channel %d, T=-%sC"%(channel,str(temp)))
                     fmin = peakmeans[0]-10
                     fmax = peakmeans[0]+10
                 else:
                     if i==0:
                         fmin = peakmeans[i]-(peakmeans[i+1]-peakmeans[i])/2.
                         lab = "initial fit"
                     else:
                         fmin = (peakmeans[i]+peakmeans[i-1])/2.
                     if i==len(peakmeans)-1:
                         fmin = peakmeans[i]-(peakmeans[i]-peakmeans[i-1])/3.
                         fmax = peakmeans[i]+(peakmeans[i]-peakmeans[i-1])/2.
                     else:
                         fmax = (peakmeans[i]+peakmeans[i+1])/2.
             #preliminary guassian fit to the peaks
                 gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
                 x = binCenters[gpts]
                 y = data[gpts]
                 [p0,p0cov] = curve_fit(gauss, xdata=x, ydata=y, p0=[150,peakmeans[i],10],bounds=([0,-100,0],[100000,200,50]))
                 subax.plot(x,gauss(x, *p0), color='orange',label=lab)
             except:
                 print("can't perform intial fit to peak #%d on channel %d"%(i,channel))
                 errfig = pl.figure()
                 errax = errfig.gca()
                 errax.hist(area,nbins,range=(lowlim,uplim))
                 errax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 errfig.show()
                 input("press Enter to continue")
                 continue
             #set the fit range based on the intial fit
             fmin = p0[1]-1*abs(p0[2])
             fmax = p0[1]+1*abs(p0[2])
             #final fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
#             gptsplot = np.logical_and((p0[1]-2*abs(p0[2]))<binCenters, (p0[1]+2*abs(p0[2]))>binCenters)
             x = binCenters[gpts]
#             xplot = binCenters[gptsplot]
             y = data[gpts]
             try:
                 [p,pcov] = curve_fit(gauss, xdata=x, ydata=y, p0=p0,bounds=([0,-100,0],[100000,200,50]))
                 perr = np.sqrt(np.diag(pcov))
                 peaklist.append(p[1])
                 peakerrlist.append(perr[1])
#                 subax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 #plot pulse area histos and fitted guassians
                 if p[1]>5:
                     lab2 = lab2+" mean=%.2f\nsigma/mean=%.2f"%(p[1],p[2]/p[1])
                 subax.plot(x,gauss(x, *p), label=lab2)
             except:
                 print("can't perform final fit to peak #%d on channel %d"%(i,channel))
                 errfig = pl.figure()
                 errax = errfig.gca()
                 errax.hist(area,nbins,range=(lowlim,uplim))
                 errax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 errfig.show()
                 input("press Enter to continue")
                 exit(0)
             subax.set_xlim([lowlim,uplim])
             subax.legend(loc="upper right")
             subax.set_title("Channel %d, T=-%s$^\circ$C"%(channel,temp),fontsize=fontsize)
             subax.set_xlabel("Pulse Area (mV*ns)",fontsize=fontsize)
             pl.savefig(data_dir%str(temp)+"Channel_%d_T-%sC.png"%(channel,str(temp)))
             pl.ion()
             subfig.show()
         input("press Enter to continue")
         pl.close()
         peaklist = np.array(peaklist)
         peakerrlist = np.array(peakerrlist)
         peakindex = np.array(list(range(len(peaklist))))
         if len(peakindex)==0:
             print ("Error: no phe peaks found on channle %d, T=%s"%(channel,temp))
             continue
         elif len(peakindex)<2:
             print ("Warning: only %d phe peak found on channel %d, T=%s"%(len(peakindex),channel,temp))
             plottemplist.append(temp)
             gainlist.append(peaklist[0])
             gainerrlist.append(peakerrlist[0])
         else:
             popt,pcov=curve_fit(linearfit,peakindex,peaklist,p0=(0.0,0.0),sigma=peakerrlist) 
             chi2 = np.sum((popt[0]*peakindex+popt[1] - peaklist) ** 2)/len(peakindex)
             if chi2>1:
                 print ("Warning: bad linear fit on channel %d, T=-%sC"%(channel,str(temp)))
             linfit = pl.figure()
             linfitax = linfit.gca()
             linfitax.set_xlabel("peak index ",fontsize=fontsize)
             linfitax.set_ylabel("pulse area (mV*ns)",fontsize=fontsize)
         #         pl.plot(peakindex,popt[0]*peakindex+popt[1],"b",label="%d*x+%d"%(popt[0],popt[1]))
             linfitax.plot(peakindex,popt[0]*peakindex+popt[1],"b",label="sphe size = %.1f$\pm$%.1f mV*ns\n chi^2/ndf=%.4f\ngain=%d"%(popt[0],np.sqrt(np.diag(pcov))[0],chi2,popt[0]*gainconv))
             linfitax.errorbar(peakindex,peaklist,yerr=peakerrlist,fmt='o',color="b")
             linfitax.set_xticks(peakindex)
             linfitax.legend(loc="lower right")
             linfitax.set_title("Channel%d, T=-%s$^\circ$C"%(channel,temp))
             linfit.show()
             linfit.savefig(data_dir%str(temp)+"gain_linearfit_ch%d.png"%(channel))
             input("press Enter to continue")
             pl.close()
             plottemplist.append(temp)
             gainlist.append(popt[0])
             gainerrlist.append(np.sqrt(np.diag(pcov))[0])
             outfile.write("%d  %f  %f\n"%(channel,gainlist[-1]*gainconv,gainerrlist[-1]*gainconv))
    #plot gain curve for the given channel
    if len(gainlist)>1:
        gainlist = np.array(gainlist)
        plottemplist = np.array(plottemplist)
        gainlist = gainlist*gainconv
        ax.errorbar((plottemplist)*(-1),gainlist,yerr=gainerrlist,fmt='o-',label="channel %d"%channel)
ax.set_xlabel("Temperature ($^\circ$C)",fontsize=fontsize)
ax.set_ylabel("SiPM Gain",fontsize=fontsize)
ax.legend(loc="upper right",prop={"size":10})
ax.set_title("Gain vs. Temperature",fontsize=fontsize)
pl.tight_layout()
#fig.savefig("/Volumes/Samsung USB/crystalize/gainvstemp.png")
fig.show()
input("press Enter to quit")

