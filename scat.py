# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:08:55 2013

@author: kronosua
"""
from pylab import *


def scat(file, start):
        dOmega=659.*494.*(7.4*10**-6)**2/(23*10**-2)**2
        a=loadtxt(file)
        a=a[a[:,0].argsort()]
        x=a[:,0]
        y=abs(a[:,1])
        x=x-x[y==y.max()]
        y=y[x>=0]
        x=x[x>=0]
        print x.max()
        t=radians(x)
        yy=y*sin(t)
        iu=interpolate.interp1d(t,yy)
        Iind=integrate.quad(iu,radians(start),radians(x.max()), epsrel=10**-4, limit=900)[0]
        I0=y.max()*dOmega/2/pi
        return Iind/(I0+Iind)*100.

def scat1(file, start, end=None, Type='linear'):
        dOmega=659.*494.*(7.4*10**-6)**2/(23*10**-2)**2
        a=loadtxt(file)
        a=a[a[:,0].argsort()]
        x=a[:,0]
        y=abs(a[:,1])
        x=x-x[y==y.max()]
        y=y[x>=0]
        x=x[x>=0]
        print x.max()
        if end is None:
                end = x.max()
        t=radians(x[x<=(end+1)])
        yy=y[x<=(end+1)]*sin(t)
        iu=interpolate.interp1d(t,yy,Type)
        Iind=integrate.quad(iu,radians(start),radians(end), epsrel=10**-4, limit=900)[0]
        I0=integrate.quad(iu,t[0],t[-1], epsrel=10**-4, limit=900)[0]     #y.max()*dOmega/2/pi
        return Iind/(I0)*100.

