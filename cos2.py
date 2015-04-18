import glob, sys
from numpy import *

fname = glob.glob(sys.argv[2])
data = loadtxt(fname[0])
x, y_meas = data.T

x=deg2rad(x)


p0 = [float(i) for i in sys.argv[1].split(',')]
print(array(p0))
A = p0[0]

def residuals(p, y, x):
     #A,
     B, C, D = p
     err = y-(A + B*cos(C*x + D)**2)
     return err

def peval(x, p):
    #A, 
    B, C, D = p
    return (A + B*cos(C*x + D)**2)

#[  8.      43.4783   1.0472]
from scipy.optimize import leastsq
plsq = leastsq(residuals, p0[1:], args=(y_meas, x))
#plsq = optimize.fmin_tnc(residuals, approx_grad=1, args=(y_meas, x), bounds=((p0[0],p0[1]), (p0[2],p0[3]), (p0[4],p0[5]), (p0[6],p0[7])),
#	x0=(p0[0]/2, p0[2]/2, p0[4]/2, p0[6]/2), epsilon=0.0001)
print(plsq)
#[ 10.9437  33.3605   0.5834]
print(array(p0))
#[ 10.      33.3333   0.5236]
#A, 
B, C, D = plsq[0]
import matplotlib.pyplot as plt
plt.rcParams['legend.fancybox'] = True
plt.scatter(x,y_meas,s=30, facecolors='none', edgecolors='k', alpha=0.3)#, label=fname[0].split("/")[-1].split(".dat")[0].replace("_fow",''))
plt.plot( x,peval(x,plsq[0]),'-w', linewidth=5,  alpha=0.5)
plt.plot( x,peval(x,plsq[0]),'--k', linewidth=4,  alpha=0.8)
#,	label='$A + B\cdot cos^2(C \cdot x + D)$\n A = %.2f \n B = %.2f \n C = %.2f \n D = %.2f '%(A,B,C,D,))
#plt.legend(shadow=True, fancybox=True)
#plt.title(r'Least-squares $cos^2$ fit')
plt.xlabel('Scattering angle, rad')
plt.ylabel('Intensity, arb. un.')
plt.grid(1)
plt.savefig(sys.argv[2]+'.png')
plt.show()