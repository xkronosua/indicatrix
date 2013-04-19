#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os
import pyfits as pf
import scipy as sp
import glob
import re
from scipy.signal import medfilt2d
import json
#import matplotlib.pyplot as plt

"""##############OPTIONS###############"""


usage = "usage: %prog {[options] args}"
parser = OptionParser(usage=usage)

parser.add_option("-f", "--filt", dest="FILTERS", help="set filters table file",\
	action='store', default="./filters.csv")

parser.add_option("-l", "--length", dest="LENGTH", help="set wave length",\
	action='store', default="532")


parser.add_option("-D", "--distance", dest="DIST", help="set sample-matrix distance, cm",\
	action='store', default="23", type='int')

parser.add_option("-N", dest="N", help="set stripes num",\
	action='store', default="1", type='int')

parser.add_option("-A", '--averaging', dest="average", help="set averaging param",\
	action='store', default="0", type='float')

parser.add_option("-t", "--type", dest="TYPE", help="set files type",\
	action='store', default="fits")

parser.add_option("-j","--journal", dest="journal", help="set journal file", action='store')

parser.add_option("-d","--dir", dest="dir", help="set input dir", action='store')
parser.add_option("-o","--out", dest="outFile", help="set output file", action='store', default="res.dat")

parser.add_option("-b","--background", dest="background", help="set background for all files", action='store', default='')
parser.add_option("--bFilt", dest="bFilt", help="set background`s filter ", action='store', default='')
parser.add_option("-z","--zero", dest="zero", help="move start", action="store_true")
parser.add_option("-p","--plot", dest="plot", help="plot data", action="store_true")
parser.add_option("-m","--medfilt", dest="medfilt", help="use median filter", action="store_true")



(options, args) = parser.parse_args()

print options, args
#################################################################
# Знайдемо кут, що захоплює матриця
MATRIX_SIZE = [494.*7.4*10**-4, 659.*7.4*10**-4]	# см
theta = sp.degrees( sp.arctan(MATRIX_SIZE[1]/2./options.DIST))

N = options.N	# кількість смуг
theta_range = None
if N == 1:
	theta_range = sp.array([0])	
else:
	theta_range = sp.linspace(-theta, theta, N)

print theta_range

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
		out.append([int(i[0]), int(i[1]), base_filt])
	return out

# Перерахунок для різних комбінацій фільтрів
def resFilters(filters, filtersDict):
	return  sp.multiply.reduce( [ filtersDict[i] for i in filters.split("+")] )

# Збір, віднімання, інтегрування, "посадження" на фільтри
def getData(Dir):
	journal = getJournal(options.journal)
	out = []
	if os.path.exists(Dir):
		
		filtersDict = getFilters(options.FILTERS, options.LENGTH)
		imgDir = os.path.join(Dir,"img")
		# Вантажимо спільний фон, якщо є
		background = None
		if options.background:
			print 'background: ' + options.background
			try: 
				background = pf.getdata(options.background)
				bpPath = options.background
				if options.bFilt:
					filt = resFilters(options.bFilt, filtersDict)
					background/=filt
					
			except IOError:
				print('BackgroundIOError')
		else:
			pass
		errList = []
		# Пробігаємось по журналу
		for degrees, minutes, filt in journal:
			pName = "^(?i)" + str(degrees) + "_" + str(minutes) + "0{,2}_"#{,1}"
			try:
				tmp = [f for f in os.listdir(Dir) if re.match(pName,f)][0]
				pPath = glob.glob(os.path.join(Dir,tmp) + '/*.' + options.TYPE)[0]
			except IndexError:
				print pName
				errList.append(pName)
				continue
			if background is None: 
				bpName = "^(?i)" + str(degrees) + "_" + str(minutes) + "0{,2}b_"#{,1}"
				try:
					tmp = [f for f in os.listdir(Dir) if re.match(bpName,f)][0]
					bpPath = glob.glob(os.path.join(Dir,tmp) + '/*.' + options.TYPE)[0]
				except IndexError:
					print bpName
					errList.append(pName)
					continue
			
			f = pf.open(pPath, mode='update')
			print pPath, f[0].header
			f[0].header.update('FILTER',filt)
			#h.update({})
			print filt#, dir(h)
			f.flush()
			#except : print "err"
			'''
			# Читаємо знімки 
			try:
				p = pf.getdata(pPath)
				print sp.shape(p)
				if background is None:
					bp = pf.getdata(bpPath)
				else:
					bp = background
				
				angle = float(degrees) + float(minutes) / 60.
				filt = resFilters(filt, filtersDict)
				t = angle + theta_range
				if options.medfilt:
					bp = medfilt2d(bp)
					p = medfilt2d(p)
				if background is None:
					signal = (p-bp)/filt*1.
				else:
					signal = (p/filt - bp)
				
				stripes = sp.array_split(signal, N, axis=1)
				print sp.shape(stripes[0])
				res = [ a.sum() for a in stripes[::-1]]
				print len(res)
				for i in range(len(res)):
					out.append([t[i], res[i]]) 
				
				# Конвертація в gif
				if options.plot:
					if not os.path.exists(imgDir):
						os.mkdir(imgDir)
					tmpName = os.path.join(Dir,"img", str(degrees) + "_" + str(minutes) + ".dat")
					print tmpName, sp.shape(signal)
					sp.savetxt(tmpName, signal)
					os.system('sh ./plotmap.sh ' + tmpName)
					os.remove(tmpName)
				
				
				
			except (IOError):	# Якщо запису в журналі не відповідають файли, то він не враховується 
				pass
				
				txt = "\033[1;31mWarning:	- "+  pPath +" | " + bpPath + "|" + str(degrees) + "_" + str(minutes) + '\033[1;m'
				errList.append(txt)
				print txt
				
			'''
		'''
		out = sp.array(out)
		if len(out)>=1 and not options.zero is None: out[:,0] = out[:,0]-(out[:,0]>180)*360
		print errList,"\nLen:\n\tjournal = " + str(len(journal)) + "\tout = " + str(len(out))
		if options.average and len(theta_range)>1:
			out = average(out, options.average*(theta_range[1] - theta_range[0]))
		'''
		return out
		
	else:
		print("Dir_error")


def getData2(inDir, bgfile):
	if os.path.exists(inDir):
		fileList = glob.glob(os.path.join(Dir,"*",'*.' + options.TYPE))
		background = pf.open(bgfile)
		bgData = background[0].data
		bgFilt = background[0].header['FILTER']


def average(data, step):
	data = data[data[:,0].argsort()]
	x = data[:,0]
	y = data[:,1]
	start = x.min()
	end = start + step
	t = True
	x_new, y_new = [], []
	while t:
		w = (x>=start) * (x< end)
		if len(x[w])>=1:
			x_new.append(x[w].mean())
			y_new.append(y[w].mean())
		start = end; end += step
		t = (start < x.max())
	return sp.array([x_new,y_new]).T
	
if __name__ == "__main__":
	#(options,args) = parser.parse_args()
	out = getData(options.dir)
	print out
	print theta_range
	
	sp.savetxt(os.path.join(options.dir,options.outFile),out)