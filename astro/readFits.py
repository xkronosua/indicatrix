#!/usr/bin/python

import pyfits as pf
import sys
import scipy as sp

def convert(fname):
	data = pf.getdata(fname)
	sp.savetxt(fname.split('.fits')[0]+".dat",data)

def integrate(fname):
	data = pf.getdata(fname)
	return data.sum()

if __name__=="__main__":
	print sys.argv
	fname = sys.argv[1]
	convert(fname)
