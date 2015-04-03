# IPython log file
'''
get_ipython().system('cat ipython_log.py')
get_ipython().magic('run ipython_log.py')
get_ipython().magic('run ipython_log.py')
ax
'''
from pylab import *

a2pb=loadtxt('sinw2_plate_b.dat')
a2p=loadtxt('sinw2_plate.dat')

a21b=loadtxt('sinw21_b.dat')
a21=loadtxt('sinw21.dat')

a22b=loadtxt('sinw22_b.dat')
a22=loadtxt('sinw22.dat')

a23b=loadtxt('sinw23_b.dat')
a23=loadtxt('sinw23.dat')

a24b=loadtxt('sinw24_b.dat')
a24=loadtxt('sinw24.dat')



ax=subplot(111,polar=1)

ax.set_rlim(0.5,10**7*2)
ax.set_rscale('log')
ax.set_rgrids(array([100,10**4,10**6]), angle=90)


w = (a2pb[:,0]<-10) + (a2pb[:,0]>10)

ax.plot(radians(a2p[:,0][::5]),a2p[:,1][::5], 'ok', mfc="w", alpha=0.9, markersize=7, markeredgewidth=0.9)
ax.plot(radians(a2pb[:,0][w][::5]),a2pb[:,1][w][::5], 'ok', mfc="w", alpha=0.9, markersize=7, markeredgewidth=0.9)
ax.plot(radians(a2pb[:,0][~w][::30]),a2pb[:,1][~w][::30], 'ok', mfc="w", alpha=0.9, markersize=7, markeredgewidth=0.9)


#w = (a21b[:,0]<-5) + (a21b[:,0]>5)
#ax.plot(radians(a21[:,0][::8]),a21[:,1][::8], '^k', alpha=0.6, markersize=7, mfc="w", markeredgewidth=0.9)
#ax.plot(radians(a21b[:,0][w][::8]),a21b[w][:,1][::8], '^k',  alpha=0.6, markersize=7, mfc="w", markeredgewidth=0.9)


w = (a22b[:,0]<-10) + (a22b[:,0]>10)
ax.plot(radians(a22[:,0][::8]),a22[:,1][::8], 'sw', alpha=0.9, markersize=7, mfc="k", markeredgewidth=0.4)
ax.plot(radians(a22b[:,0][w][::8]),a22b[w][:,1][::8], 'sw',  alpha=0.9, markersize=7, mfc="k", markeredgewidth=0.4)

ax.plot(radians(a22b[:,0][~w][::50]),a22b[~w][:,1][::50], 'sw', alpha=0.9, markersize=7, mfc="k", markeredgewidth=0.4)

#w = (a23b[:,0]<-5) + (a23b[:,0]>5)
#ax.plot(radians(a23[:,0][::8]),a23[:,1][::8], 'sk',  alpha=0.9, markersize=7, mfc="w", markeredgewidth=0.7)
#ax.plot(radians(a23b[:,0][w][::8]),a23b[:,1][w][::8], 'sk',  alpha=0.9, markersize=7, mfc="w", markeredgewidth=0.7)



w = (a24b[:,0]<-10) + (a24b[:,0]>10)
ax.plot(radians(a24[:,0][::8]),a24[:,1][::8], 'ok',  alpha=0.9, markersize=9, mec="w", markeredgewidth=0.4)
ax.plot(radians(a24b[:,0][w][::8]),a24b[:,1][w][::8], 'ok',  alpha=0.9, markersize=9, mec="w", markeredgewidth=0.4)
ax.plot(radians(a24b[:,0][~w][::15]),a24b[:,1][~w][::15], 'ok',  alpha=0.9, markersize=9, mec="w", markeredgewidth=0.4)
rcParams.update({'font.size': 18})
show()