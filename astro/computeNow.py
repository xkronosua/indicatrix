#!/usr/bin/python

import scipy as sp
import pyfits as pf
from optparse import OptionParser
import os


usage = "usage: %prog [option] arg [[option] arg ...]"
parser = OptionParser(usage=usage)
parser.add_option("-s", "--signal", action="store", dest="signal", help="set signal's file")
parser.add_option("-b", "--bg", action="store", dest="background", help="set background's file")
parser.add_option("-a", "--angle", action="store", dest="angle", help="set angle in format ANGLE_MINUTES:\n%prog -a 240_30")
parser.add_option("-d", "--dir", action="store", dest="dir", help="set root dir", default="./")

def anglesSearch():
	data = []
	for root, dirs, files in os.walk(options.dir):
		for file in files:		
			arr = file.split("_")
			if len(arr)<2: break
			if arr[0] + "_" + arr[1] == options.angle or \
				arr[0] + "_" + arr[1] == options.angle + "b":
					data.append(os.path.join(root,file))
					print(root+"/"+file)
			else: pass #print("No such files in"+root)
	if len(data) == 2:
		s1 = pf.getdata(data[0])
		s2 = pf.getdata(data[1])
		res = abs(s2-s1).sum()
		print("\n\tSUM = %f" % res)
	else: print("No such files")
		
if __name__ == "__main__":
	(options, args) = parser.parse_args()
	if options.angle:
		anglesSearch()
	elif any(args) and not options.angle:
		
		#print args, options.angle
		options.angle = args[0]
		#print options.angle
		anglesSearch()
				
	elif options.signal and options.bg:
		s2 = pf.getdata(options.signal)
		s1 = pf.getdata(options.bg)
		res = abs(s2-s1).sum()
		print("\tSUM = ",res)
	
	else: print("try -h or --help for help")
	
