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
	if options.dir:
		listdir = commands.getstatusoutput('find ' + options.dir + ' -name "*"')[1].split("\n")	#os.listdir(options.dir)
		
		names = []
		out = []
		for i in listdir:
			#print i
			if os.path.isfile(i) and i.split('.')[-1] == "fits":
				for j in listdir:
					#print j
					t = j.split('/')[-1]
					if  (t[:-11]+t[-10:]) == i.split("/")[-1]:
						#print t
						names.append((i,j))
		print  len(names)
		for i, ii in names:
			#print i,ii
			print i.split("/")[-1]
			try:
				s = pf.getdata(i)
				
				#ii = i[:-10] + "b" + i[-10:]	
				b = pf.getdata(ii)
				#print b
				res = (s-b).sum()
				temp = i.split("/")[-1].split('.fits')[0].split("_")
				#print temp
				angle = float(temp[-3]) + float(temp[-2])/60.
				out.append( [ angle, res ] )
			except IOError:
				
				
				print ("warning: \t%s\t%s" % (i, ii))
				
		sp.savetxt( os.path.join(options.dir,options.out), sp.array(out) )
		
