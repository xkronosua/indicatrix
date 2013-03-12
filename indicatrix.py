#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os
import pyfits as pf
import scipy as sp
import glob

#import matplotlib.pyplot as plt

"""##############OPTIONS###############"""

usage = "usage: %prog {[options] args}"
parser = OptionParser(usage=usage)

parser.add_option("-f", "--filt", dest="FILTERS", help="set filters table file",\
	action='store', default="./filters.csv")

parser.add_option("-l", "--length", dest="LENGTH", help="set wave length",\
	action='store', default="532")

parser.add_option("-t", "--type", dest="TYPE", help="set files type",\
	action='store', default="fits")

parser.add_option("-j","--journal", dest="journal", help="set journal file", action='store')

parser.add_option("-d","--dir", dest="dir", help="set input dir", action='store')
parser.add_option("-o","--out", dest="outFile", help="set output file", action='store', default="res.dat")

parser.add_option("-b","--background", dest="background", help="set background for all files", action='store', default='')
parser.add_option("-z","--zero", dest="zero", help="move start", action="store_true")
parser.add_option("-p","--plot", dest="plot", help="plot data", action="store_true")



(options, args) = parser.parse_args()

print options, args

# Читання таблиці фільтрів та вибір значень для даної довжини хвилі
def getFilters(file="./filters.csv", length="532"):
	filt = sp.loadtxt(file, dtype="S")
	col = sp.where(filt[0,:]==length)[0][0]
	return dict( zip(filt[1:,0], sp.array(filt[1:,col],dtype="f") ) )
# Читання журналу
def getJournal(file):
	base_angle, base_filt = ['']*2
	try:
		data = sp.loadtxt(file,dtype="S")
	#	print data
		base_angle = data[0,0]
	except (IndexError,ValueError):
		data = sp.loadtxt(file,dtype="S",delimiter=',')
		base_angle = data[0,0]
		base_filt = data[0,2]
	out = []
	
	for i in data:
		# Якщо кути в таблиці вводяться лише раз для групи мінут
		if i[0] == "":
			i[0] = base_angle
		else: base_angle = i[0]
		if i[2] == "": 
			i[2] = base_filt
		else: base_filt = i[2]
		out.append([i[0] + "_" + i[1], i[2]])
	return out

# Перерахунок для різних комбінацій фільтрів
def resFilters(filters, filtersDict):
	return  sp.multiply.reduce( [ filtersDict[i] for i in filters.split("+")] )

# Збір, віднімання, інтегрування, "посадження" на фільтри
def getData(dir):
	journal = getJournal(options.journal)
	out = []
	if os.path.exists(dir):
		#os.mkdir(os.path.join(dir,'img'))
		filtersDict = getFilters(options.FILTERS, options.LENGTH)
		names = []
		for root, dirs, files in os.walk(options.dir):
			for file in files:
				if file.split(".")[-1] == options.TYPE:
					names.append([root,file])
				else: pass
		
		for angle_name, filt in journal:
			
			pName, bpName = ['']*2
			
			for root, file in names:
				dir = os.path.basename(root) #----
				ang = dir.split("_")
				if ang[0] + "_" + ang[1] == angle_name or  ang[0] + "_" + ang[1] == angle_name + "0":
					pName = os.path.join(root, file)
				elif options.background:
					bpName = options.background
				elif ang[0] + "_" + ang[1] == angle_name+"b" or  ang[0] + "_" + ang[1] == angle_name+"0b":
					bpName = os.path.join(root, file)
				else: pass
				
			try:
				p = pf.getdata(pName)
				bp = pf.getdata(bpName)
				ang = angle_name.split("_")
				angle = float(ang[0]) + float(ang[1]) / 60.
				filt = resFilters(filt, filtersDict)
				if not options.plot is None:
					tmpName = os.path.split(pName)[0] + '/' + str(round(angle,3)) + '.dat'
					print tmpName
					sp.savetxt(tmpName, (p-bp)/filt)
					os.system('sh ./plotmap.sh ' + tmpName)
					os.remove(tmpName)
				out.append([ angle, abs(p-bp).sum() /filt ])
			except (IOError, ValueError, IndexError):	# Якщо запису в журналі не відповідають файли, то він не враховується 
				print("\033[1;31mWarning:	- "+  pName +" | " + bpName + "|" + angle_name + '\033[1;m') 
		out = sp.array(out)
		if len(out)>=1 and not options.zero is None: out[:,0] = out[:,0]-(out[:,0]>180)*360
		print("Len:\n\tjournal = " + str(len(journal)) + "\tout = " + str(len(out)))
		return out
		
	else:
		print("Dir_error")

if __name__ == "__main__":
	(options,args) = parser.parse_args()
	out = getData(options.dir)
	print out
	sp.savetxt(os.path.join(options.dir,options.outFile),out)
