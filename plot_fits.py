# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:17:58 2013

@author: kronosua
"""

#!/usr/bin/python

import pyfits as pf
import sys
import os
import scipy as sp

def convert(fname):
	data = pf.getdata(fname)
	name = fname.split('.fits')[0]+".dat"
	sp.savetxt(name,data)
	return name

def integrate(fname):
	data = pf.getdata(fname)
	return data.sum()



if __name__=="__main__":
	print sys.argv
	fname = sys.argv[1]
	name = convert(fname)
	os.system('sh ./plotmap.sh '+ name)
 