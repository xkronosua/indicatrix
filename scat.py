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



def scat(file_path, cross_path, start, end=None, Type='nearest'):
		dOmega=4*sp.pi*sp.sin(sp.radians(1.38/2))**2#0.36*0.49*10**-6/(39*10**-2)**2#659.*494.*(7.4*10**-6)**2/(23*10**-2)**2
		print(dOmega)

		a=sp.loadtxt(file_path)
		a=a[a[:,0].argsort()]
		c=sp.loadtxt(cross_path)
		c=c[c[:,0].argsort()]
		'''		
		x1=c[:,0]
		y1=abs(c[:,1])
		x1=x1-x1[y1==y1.max()]
		y1=y1[x1>=0]
		x1=x1[x1>=0]
		print( x1.max())
		#if end1 is None:
		#		end1 = x1.max()
		t1=sp.radians(x1[x1<=(end+1)])
		yy1=y1[x1<=(end+1)]*sp.sin(t1)
		iu1=interp1d(t1,yy1/y1.max(),Type)

		Iind1=quad(iu1,sp.radians(start),t1.max(), epsrel=10**-4, limit=900)[0] + quad(iu1,sp.radians(0),sp.radians(start), epsrel=10**-4, limit=400)[0]
		

		x1=c[:,0]
		y1=abs(c[:,1])
		x1=x1-x1[y1==y1.max()]
		y1=y1[x1<=0]
		x1=x1[x1<=0]
		x1=abs(x1)
		print( x1.max())
		#if end1 is None:
		#		end1 = x1.max()
		t1=sp.radians(x1[x1<=(end+1)])
		yy1=y1[x1<=(end+1)]*sp.sin(t1)
		iu1=interp1d(t1,yy1/y1.max(),Type)

		Iind1_=quad(iu1,sp.radians(start),t1.max(), epsrel=10**-4, limit=900)[0] + quad(iu1,sp.radians(0),sp.radians(start), epsrel=10**-4, limit=400)[0]
		'''		
		x1=c[:,0]

		y1=abs(c[:,1])
		y_max = y1.max()
		y1 /= y_max
		w1 = (x1<8)*(x1>-8)
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
		iu1=interp1d(t1,yy1,Type)
		plt.semilogy(t1,yy1,'.b', t1, iu1(t1),'k')
		#plt.show()
		Iind1_abs=quad(iu1,sp.radians(1.5),t1.max(), epsrel=10**-4, limit=900)[0] + quad(iu1,t1.min(),sp.radians(1.5), epsrel=10**-4, limit=400)[0]
		
		'''

		x=a[:,0]
		y=abs(a[:,1])
		#x=x-x[y==y.max()]
		y=y[x>=0]
		x=x[x>=0]
		print( x.max())
		if end is None:
				end = x.max()
		t=sp.radians(x[x<=(end+1)])
		yy=y[x<=(end+1)]*sp.sin(t)
		iu=interp1d(t,yy/y1.max(),Type)
		Iind=quad(iu,sp.radians(start),sp.radians(end), epsrel=10**-4, limit=900)[0]
		x=a[:,0]
		y=abs(a[:,1])
		#x=x-x[y==y.max()]
		y=y[x<0]
		x=x[x<0]
		x=abs(x)
		print( x.max())
		if end is None:
				end = x.max()
		t=sp.radians(x[x<=(end+1)])
		yy=y[x<=(end+1)]*sp.sin(t)
		iu=interp1d(t,yy/y1.max(),Type)
		Iind_=quad(iu,sp.radians(start),sp.radians(end), epsrel=10**-4, limit=900)[0]
		'''
		x=a[:,0]
		y=abs(a[:,1])
		y /= y_max
		w = (x<7)*(x>-7)
		m = leastsq(residuals,[0,0,1.],args=(y[w],x[w]),full_output=1)
		print("center: ", m[0])
		x-=m[0][1]
		#x=x-x[y==y.max()]
		#y=y[x>0]
		#x=x[x>0]
		
		x=abs(x)
		y = y[x.argsort()]
		x = x[x.argsort()]
		print( x.max())
		if end is None:
				end = x.max()
		t=sp.radians(x)
		yy=y*sp.sin(t)
		iu=interp1d(t,yy,Type)
		plt.semilogy(t,yy,'xm', t, iu(t),'r',[sp.radians(start)]*200, sp.linspace(0,yy.max(),200),'r-')
		plt.show()
		Iind_abs=quad(iu,sp.radians(start),sp.radians(end), epsrel=10**-4, limit=900)[0]
		#I0=integrate.quad(iu,t[0],t[-1], epsrel=10**-4, limit=900)[0]

		#I0=integrate.quad(iu,t[0],t[-1], epsrel=10**-4, limit=900)[0]	 #y.max()*dOmega/2/pi
		#l_abs, l_parts = Iind_abs/(Iind1_abs), (Iind+Iind_)/(Iind1+Iind1_)

		return  Iind_abs/(Iind1_abs)

if __name__ == "__main__":
	print(scat(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))	