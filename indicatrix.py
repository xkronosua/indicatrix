#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os
import scipy as sp
import matplotlib.pyplot as plt

"""##############OPTIONS###############"""

usage = "usage: indicatrix.py {[options] args}"
parser = OptionParser(usage=usage)

parser.add_option("-f", "--filt", dest="FILTERS", help="set filters table file",\
	action='store', default="./filters.csv")

parser.add_option("-l", "--length", dest="LENGTH", help="set wave length",\
	action='store', default="532")

parser.add_option("--left", dest="Left", help="set journal for left side", action='store')
parser.add_option("--cleft", dest="cLeft", help="set journal for left cross", action='store')
parser.add_option("--right", dest="Right", help="set journal for right side", action='store')
parser.add_option("--cright", dest="cRight", help="set journal for right cross", action='store')

parser.add_option("--dirL", dest="dirLeft", help="set input dir for Left side", action='store')
parser.add_option("--cdirL", dest="cdirLeft", help="set cross input dir Left side", action='store')
parser.add_option("--dirR", dest="dirRight", help="set input dir Right side", action='store')
parser.add_option("--cdirR", dest="cdirRight", help="set cross input dir Right side", action='store')
parser.add_option("--out", dest="outDir", help="set output dir", action='store', default="./")


(options, args) = parser.parse_args()

print options, args

# Читання таблиці фільтрів та вибір значень для даної довжини хвилі
def getFilters(file="./filters.csv", length="532"):
	filt = sp.loadtxt(file, dtype="S")
	col = sp.where(filt[0,:]==length)[0][0]
	return dict( zip(filt[1:,0], sp.array(filt[1:,col],dtype="f") ) )
# Читання журналу
def getJournal(file):
	data = sp.loadtxt(file,dtype="S")
	angles = sp.array( data[:,2],dtype="f") +  sp.array( data[:,3],dtype="f")/60.
	return sp.array(data[:,0],dtype="i"), data[:,1], angles

# Інтегрування методом Сімпсона (x[i+1]-x[i]=1)
def Int(f):
	return sum( [ (f[i]+f[i+1])/2. for i in range(len(f)-1)])
# Перерахунок для різних комбінацій фільтрів
def resFilters(filters, filtersDict):
	return  sp.multiply.reduce( [ filtersDict[i] for i in filters] )
# Збір, віднімання, інтегрування, "посадження" на фільтри
def getData(dir, journal):
	out = []
	if os.path.exists(dir):
		filtersDict = getFilters(options.FILTERS, options.LENGTH)
		for I in zip(journal[0],journal[1],journal[2]):
			i, filt, angle = I
			pName = os.path.join(dir,"1P" + str(i) + ".DAT")
			bpName = os.path.join(dir,"1BP" + str(i) + ".DAT")
			try:
				p = sp.loadtxt(pName,dtype='f')
				bp = sp.loadtxt(bpName,dtype='f')
				out.append([i, angle, Int(p[:,1]-bp[:,1]) / resFilters(filt, filtersDict) ])
			except IOError:	# Якщо запису в журналі не відповідають файли, то він не враховується 
				print("warning:	- ", i, pName, bpName) 
		return sp.array(out)
	else:
		print("Dir_error")

# Ручна обрізка по кросу
def centre_cut(XY, c_XY, s):
	print("centre_cut")
	out = ""
	if  s=="l":
		x, y, xx, yy = XY[:,0], XY[:,1], c_XY[:,0], c_XY[:,1]#load_left(path)
		out = "LEFT_CENTRED.DAT"
	else:
		x, y, xx, yy = XY[:,0], XY[:,1], c_XY[:,0], c_XY[:,1]#load_right(path)
		out = "RIGHT_CENTRED.DAT"
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.plot(x,y,'bo',xx,yy,'go')
	x_new = sp.array([])
	y_new = sp.array([])
	
	def onclick(event):
		if event.button==3:
			plt.close()
		else:
			ax.plot(x,y,'bo',xx,yy,'go')
			plt.hold(True)
			diff = abs(x-event.xdata)
			del_from = sp.where(diff==diff.min())[0]
			print del_from
			if s=="l":
				x_new = x[del_from:]
				y_new = y[del_from:]
			else:
				x_new = x[:del_from]
				y_new = y[:del_from]

			ax.plot(x[del_from],y[del_from],'rx',markersize=20)
			plt.draw()
			plt.hold(False)
			print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
			sp.savetxt(os.path.join(options.outDir,out),sp.array([x_new,y_new]).T)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
################################################
sideType = ""
out, c_out= [], []
# Якщо задані теки та файли для певної частини, то вони й рахуються :) 
if options.Left and options.dirLeft and options.cLeft and options.cdirLeft:
	journal = getJournal(options.Left)
	out = getData(options.dirLeft, journal)
	c_journal = getJournal(options.cLeft)
	c_out = getData(options.cdirLeft, c_journal)
	sideType = "l"
	
elif options.Right and options.dirRight and options.cRight and options.cdirRight:
	journal = getJournal(options.Right)
	out = getData(options.dirRight, journal)
	c_journal = getJournal(options.cRight)
	c_out = getData(options.cdirRight, c_journal)
	sideType = "r"
else: print "error :)"

print sp.shape(out), sp.shape(c_out)
centre_cut(out[:,1:], c_out[:,1:], sideType)
