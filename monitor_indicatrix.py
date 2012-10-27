#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os, time, sys
import pyfits as pf
import scipy as sp
import warnings
warnings.resetwarnings()
#warnings.filterwarnings('error', category=UserWarning, append=True)
#from subprocess import call
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

parser.add_option("-s", "--sleep", dest="SLEEP", help="set sleep time",\
action='store', default=1)

parser.add_option("-j","--journal", dest="journal", help="set journal file", action='store')

parser.add_option("-d","--dir", dest="dir", help="set input dir", action='store')
parser.add_option("-o","--out", dest="outFile", help="set output file", action='store', default="res.dat")
parser.add_option("-p","--plot", dest="plot", help="plot output data", action='store_true', default=False)


(options, args) = parser.parse_args()



# Читання таблиці фільтрів та вибір значень для даної довжини хвилі
def getFilters(file="./filters.csv", length = "532"):
	filt = sp.loadtxt(file, dtype="S")
	col = sp.where(filt[0,:]==length)[0][0]
	print col
	return dict( zip(filt[1:,0], sp.array(filt[1:,col],dtype="f") ) )

# Читання журналу
def getJournal(file):
	base_angle, base_filt = ['']*2
	try:
		data = sp.loadtxt(file,dtype="S")
		base_angle = data[0,0]
	except IndexError:
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
	

# Перевірка на наявність нових або змінених знімків
def getDirUpdates(timestamp):
	new = []
	all = []
	for root, dirs, files in os.walk(options.dir):
		for basename in files:
			if os.path.splitext(basename)[1] == "."+options.TYPE:
				filename = os.path.join(root, basename)
				all.append(filename)
				status = os.stat(filename)
				if status.st_mtime > timestamp:
					#print "\t%10s\t\t+" % (filename)
					new.append(filename)
				else:
					pass
					#print "\t%10s\t\t-" % (filename)
	return new, all


if __name__ == "__main__":
	(options,args) = parser.parse_args()
	timestamp = 0
	out_array = sp.zeros((0,2))
	if os.path.exists(options.dir) and os.path.exists(options.journal):
		print( options)
		filtersDict = getFilters(options.FILTERS, options.LENGTH)
	else:
		sys.exit("No base files")
	# Читання попередніх даних, якщо такі є
	outfile = os.path.join(options.dir,options.outFile)
	try:
		out_array = sp.loadtxt(outfile)
	except IOError: print("No oldest data")

	while True:
		try:
			new, all = getDirUpdates(timestamp)
			if any(new):
				#############################
				journal = getJournal(options.journal)
				out = []
				for angle_name, filt in journal:
			
					pName, bpName = ['3','3'] 
					
					for file in new:
						ang = os.path.basename(file).split("_")
						
						if ang[0] + "_" + ang[1] == angle_name or  ang[0] + "_" + ang[1] == angle_name + "0":
							pName = file
						elif ang[0] + "_" + ang[1] == angle_name+"b" or  ang[0] + "_" + ang[1] == angle_name+"0b":
							bpName = file
						else: pass
					
					try:
						p = pf.getdata(pName)
						bp = pf.getdata(bpName)
						ang = angle_name.split("_")
						angle = float(ang[0]) + float(ang[1]) / 60.
						out.append([ angle, abs(p-bp).sum() / resFilters(filt, filtersDict) ])
					except (IOError ,ValueError):	# Якщо запису в журналі не відповідають файли, то він не враховується 
						pass
						#print("\033[1;31mWarning:	- "+  pName +" | " + bpName + "|" + angle_name + '\033[1;m') 
				out = sp.array(out)
				out[:,0] = out[:,0]-(out[:,0]>180)*360	# Якщо кути більші за 180, то переводимл у від’ємні
				print("Len:\n\tjournal = " + str(len(journal)) + "\tout = " + str(len(out)))
				
				out[out[:,0].argsort()]
				for i, new_out in enumerate(out[:,0]):
					if sp.any(out_array[:, 0] == new_out):
						print("\t%10s\t\t-\tupdated" % (str(new_out)))
						out_array = out_array[ abs(out_array[:,0] - new_out)>10**-3 ]
				out_array = sp.concatenate((out_array,out))
				sp.savetxt(outfile,out_array)

				#############################
			else:
				print("waiting...")
			timestamp = time.time()
			if options.plot:
				os.system("./plot.sh "+outfile)
			time.sleep(options.SLEEP)
			
		except KeyboardInterrupt:
			sys.exit(0)
	
