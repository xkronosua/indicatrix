#!/usr/bin/python

import pyfits as pf
import scipy as sp
import os
import commands
from optparse import OptionParser



usage = "usage: %prog [options] arg1 arg2"
parser = OptionParser(usage=usage)
parser.add_option("-d", "--dir", action="store", dest="dir", default="./", help="set inFiles dir")
parser.add_option("-o", "--out", action="store", dest="out", default="res.dat", help="set out file name. default: res.dat")


if __name__ == "__main__":
	(options, args) = parser.parse_args() 
	if options.dir[-1] != "/": options.dir+="/"
	print args
	if os.path.exists(options.dir):
		listdir = commands.getstatusoutput('find ' + options.dir + ' -name "*.fits"')[1].split("\n")	#os.listdir(options.dir)
		
		names = []
		out = []
		for i in listdir:
			for j in listdir:
				t = j.split('/')[-1]
				if  (t[:-11]+t[-10:]) == i.split("/")[-1]:
					names.append((i,j))
		print("#",len(names))
		for i, ii in names:
			print i.split("/")[-1]
			try:
				s = pf.getdata(i)
				b = pf.getdata(ii)
				res = abs((s-b)).sum()
				temp = i.split("/")[-1].split('.fits')[0].split("_")
				angle = float(temp[-3]) + float(temp[-2])/60.
				out.append( [ angle, res ] )
			except IOError:
				print ("warning: \t%s\t%s" % (i, ii))
		out = sp.array(out)
		out = out[ out[:,0].argsort() ]
		sp.savetxt( os.path.join(options.dir,options.out), out)
		
