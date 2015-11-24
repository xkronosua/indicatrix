import scipy as sp
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import leastsq # Levenberg-Marquadt Algorithm #

def lorentzian(x,p):
    numerator =  (p[0]**2 )
    denominator = ( x - (p[1]) )**2 + p[0]**2
    y = p[2]*(numerator/denominator)
    return y

def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err

k_1064 = 4.31611559639e-12#
k_532 = 1.65375545555e-14#4.5207341216591067e-15
K = {"532":k_532, "1064":k_1064}
def scat(file_path, cross_path, wavelength='532', start=1.5, end=None, part=None, Type='nearest'):
		dOmega = (7.4*10**-4)**2*4/(39.2)**2#4*sp.pi*sp.sin(sp.radians(1.38/2))**2#0.36*0.49*10**-6/(39*10**-2)**2#659.*494.*(7.4*10**-6)**2/(23*10**-2)**2
		print(dOmega)
		print("file_path=", file_path)
		a=sp.loadtxt(file_path)
		a=a[a[:,0].argsort()]
		#a[:,1] /= dOmega
		print("cross_path=", cross_path)
		print("Cross", cross_path)
		c=sp.loadtxt(cross_path)
		c=c[c[:,0].argsort()]
		#c[:,1] /= dOmega
		
		x1=c[:,0]

		y1=abs(c[:,1])
		print("Cross_max=", max(y1))
		y1_max = y1.max()
		x1=x1-x1[y1==y1.max()][0]
		y1 /= y1_max
		w1 = (x1<20)*(x1>-20)
		m = leastsq(residuals,[0,0,1],args=(y1[w1],x1[w1]),full_output=1)
		print("center: ", m[0])
		x1-=m[0][1]
		#y1=y1[x1>=0]
		#x1=x1[x1>=0]
		x1=abs(x1)
		y1 = y1[x1.argsort()]
		x1 = x1[x1.argsort()]
		print( x1.max())
		#if end1 is None:
		#		end1 = x1.max()
		t1=sp.radians(x1)
		yy1=y1*sp.sin(t1)
		yy1_max = yy1.max()
		#yy1 /= yy1_max
		iu1=interp1d(t1,yy1,Type)
		plt.semilogy(t1,yy1,'.b', t1, iu1(t1),'k')
		
		
		x=[]
		if a[:,0].max()<60:
			x= -a[:,0]
		else:
			x=a[:,0]
		y=abs(a[:,1])
		y_max = y.max()
		x=x-x[y==y.max()][0]
		y /= y_max
		w = (x<20)*(x>-20)
		m = leastsq(residuals,[0,0,1.],args=(y[w],x[w]),full_output=1)
		print("center: ", m[0])
		x-=m[0][1]
		
		if part == "left":
			y=y[x>0]
			x=x[x>0]
		elif part == "right":
			y=y[x<0]
			x=x[x<0]

		x=abs(x)
		y = y[x.argsort()]
		x = x[x.argsort()]
		print( x.max())
		if end is None:
				end = x.max()
		t=sp.radians(x)
		yy=y*sp.sin(t)
		yy_max = yy.max()
		#yy /= yy_max
		iu=interp1d(t,yy,Type)
		plt.semilogy(t,yy,'xm', t, iu(t),'r',[sp.radians(start)]*200, sp.linspace(0,yy.max(),200),'r-',
			[sp.radians(end)]*200, sp.linspace(0,yy.max(),200),'g-')
		plt.show()
		print(yy.max(), yy.min())
		Iind_abs=quad(iu,sp.radians(start),sp.radians(end), epsrel=10**-8, limit=1000)[0]
		p_scat = 2*sp.pi*Iind_abs  * y_max / dOmega * K[wavelength]
		print('p_scat = ', p_scat)
		#I0=integrate.quad(iu,t[0],t[-1], epsrel=10**-4, limit=900)[0]

		#I0=integrate.quad(iu,t[0],t[-1], epsrel=10**-4, limit=900)[0]	 #y.max()*dOmega/2/pi
		#l_abs, l_parts = Iind_abs/(Iind1_abs), (Iind+Iind_)/(Iind1+Iind1_)

		#return  Iind_abs/(Iind1_abs)

if __name__ == "__main__":
	scat(sys.argv[2], sys.argv[1], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]) , sys.argv[6])
