#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os
import pyfits as pf
import scipy as sp
import glob
import re
from scipy.signal import medfilt2d
import scipy.optimize as optimize
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import json
import traceback
from scipy import ndimage
from pylab import *
#import matplotlib.pyplot as plt



##### sorting names #####

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

##### sorting names #####

"""##############OPTIONS###############"""


usage = "usage: %prog {[options] args}"
parser = OptionParser(usage=usage)

parser.add_option("-f", "--filt", dest="FILTERS", help="set filters table file",\
	action='store', default="./filters.dict")

parser.add_option("-l", "--length", dest="LENGTH", help=u"set wave length",\
	action='store', default="532")


parser.add_option("-D", "--distance", dest="DIST", help="set sample-matrix distance, cm",\
	action='store', default="23", type='float')

parser.add_option("-N", dest="N", help="set stripes num",\
	action='store', default="330", type='int')

parser.add_option("-s", dest="slice", help="cut subrect ",\
	action='store', default="")

parser.add_option("-A", '--averaging', dest="average", help="set averaging param",\
	action='store', default="0", type='float')

parser.add_option("-t", "--type", dest="TYPE", help="set files type",\
	action='store', default="fits")

parser.add_option("-j","--journal", dest="journal", help="set journal file", action='store')

parser.add_option("-d","--dir", dest="dir", help="set input dir", action='store')
parser.add_option("-o","--out", dest="outFile", help="set output file", action='store', default="res.dat")

parser.add_option("-b","--background", dest="background", help="set background for all files", action='store', default='')
parser.add_option("--bFilt", dest="bFilt", help="set background`s filter ", action='store', default='')
parser.add_option("--bInterp", dest="bInterp", help="interpolate background`s", action='store_true')
parser.add_option("-z","--zero", dest="zero", help="move start", action="store_true")
parser.add_option("-p","--plot", dest="plot", help="plot data", action="store_true")
parser.add_option("-P","--panorama", dest="panorama", help="create panorama", action="store_true")
parser.add_option("-m","--medfilt", dest="medfilt", help="use median filter", default=0, type='int', action="store")
parser.add_option("-g","--gaussfilt", dest="gaussfilt", help="use gauss filter", default='1x1', action="store")
parser.add_option("--bounds", dest="bounds", help="cut min and max", default='x', action="store")
parser.add_option("--optimBounds", dest="optimBounds", help="optimization bounds", default='-10,10,0,50', action="store")


(options, args) = parser.parse_args()

print( options, args)

#print theta_range
def cut_minmax(data, bounds):
	if len(bounds) != 0:
		
		data *= (data>bounds[0])*(data<bounds[1])
	return data

def filtCalc(filters, filtTable=None):
	if filters:
		if not filtTable is None:
			filters = filters.replace(' ', '').replace('+', ',').replace(';', ',').replace('.', ',')
			res = 1.
			try:
				res = sp.multiply.reduce( [ filtTable[options.LENGTH][i.upper()] for i in filters.split(",")] )
			except KeyError:
				res = 1.
			return res
		else:
			return	
	else:
		return 1.


def getData(Dir,  bgfile=''):
	#################################################################

	try:
		bounds = [int(i) for i in options.bounds.split('x')]
	except ValueError:
		bounds = []

	# Знайдемо кут, що захоплює матриця
	MATRIX_SIZE = [494.*7.4*10**-4, 659.*7.4*10**-4]	# см
	theta = sp.degrees( sp.arctan(MATRIX_SIZE[1]/options.DIST))/2.
	print( "theta: %f"%theta)
	N = options.N	# кількість смуг
	theta_range = None
	if N == 1:
		theta_range = sp.array([0])	
	else:
		theta_range = sp.linspace(-theta, theta, N)

	if os.path.exists(Dir):
		filtTable = json.load(open(options.FILTERS))
		
		# background
		bOut = []
		data = []
		if bgfile:
			if options.bInterp:
				backgrounds = glob.glob(bgfile)
				
				print(backgrounds)
				for i in backgrounds:
					print(i)
					background = pf.open(i)[0]
					data = background.data
					print( sp.shape(data))
					if 'FILTER' in  background.header.keys():
						bgFilt = filtCalc(background.header['FILTER'], filtTable)
					else:
						bgFilt = 1.
					data = data / bgFilt
					bname = background.header['OBJECT']
					print(bname)
					angle = float(bname[1:].split('_')[0])
					if options.zero:
						angle -= (angle>180)*360
					bOut.append([angle, data.mean()])

				bOut = sp.array(bOut)
				bOut = bOut[bOut[:,0].argsort()]
				#plt.plot(bOut[:,0],bOut[:,1])
				#plt.show(False)
				bgData = sp.poly1d(sp.polyfit(bOut[:,0],bOut[:,1], 5))
				plot(bOut[:,0],bOut[:,1])
				show()
			else:
				background = pf.open(glob.glob(bgfile)[0])[0]
				bgData = background.data
				print( sp.shape(bgData))
				if 'FILTER' in  background.header.keys():
					bgFilt = filtCalc(background.header['FILTER'], filtTable)
				else:
					bgFilt = 1.
				bgData = bgData / bgFilt
		else:
			print('No BG')
			pass
		sh = sp.shape(bgData)
		#if options.medfilt:
		#	bgData = medfilt2d(bgData)

		fileList = glob.glob(os.path.join(Dir,"*",'*.fits'))
		out = []
		sort_nicely(fileList)
		
		for f in fileList:
			profData = None
			print(f)
			try:
				prof = pf.open(f)
				profData = prof[-1].data
				if 'FILTER' in prof[-1].header.keys():
					print( prof[-1].header['FILTER'])
					profFilt = filtCalc(prof[-1].header['FILTER'], filtTable)
				else:
					profFilt = 1.

				profData = cut_minmax(profData, bounds)
				profName = prof[-1].header['OBJECT']
				profData = profData / profFilt
			except (IOError, ValueError, IndexError):
				traceback.print_exc()
				continue

			
			try:
				degrees, minutes = profName.split('_')
				angle = float(degrees) + float(minutes) / 60.
				if options.zero:
					angle -= (angle>180)*360
			except ValueError:
				traceback.print_exc()
				continue
			signal = []
			if options.bInterp:
				try:
					
					signal = (profData - bgData(angle))
					print(signal.min())	
					print('='*50)
				except:
					traceback.print_exc()
					print(angle)
					print(sp.mean(bOut[:,1]), type(bOut))
					print( angle, bOut[:,0])
					try:
						signal = abs(profData - sp.mean(bOut[:,1]))
					except:
						traceback.print_exc()
					print(signal.min())
			else:
				signal = profData - bgData.mean()
			signal = signal - signal*(signal<0)
			# medfilt
			gf = [ int(i) for i in options.gaussfilt.split('x')]
			signal = ndimage.gaussian_filter(signal, gf)

			if options.medfilt:
				signal = medfilt2d(signal, options.medfilt)
			
			if options.panorama:
				theta_range = sp.linspace(-theta, theta, sp.shape(signal)[1])
			
			
			
			t = angle + theta_range[::-1]
			out.append([t, signal])

	#	if len(out)>=1 and not options.zero is None: out[:,0] = out[:,0]-(out[:,0]>180)*360
	return out
"""
def shift(out1, out2):
	x1, data1 = out1
	x2, data2 = out2
	w1 = where(x2<=x1.max())[0]
	w2 = where(x1>=x2.min())[0]
	l = len(w1)//2

	errorfunction = lambda p: sum(abs(data1[:,w1[l:]]-data2[:,w2[:-l] + int(p)]))
	params = (1,)
	p, success = optimize.leastsq(errorfunction, params)


def diff(p):
	return abs(p[1]-p[0]).sum()
"""
b = [int(i) for i in options.optimBounds.split(',')]
optimBounds = (b[0],b[1]),(b[2],b[3])


def test(n1,n2,eps=0.5, show=True, returnAll=False):
	x1, data1 = OUT['res'][n1]
	x2, data2 = OUT['res'][n2]
	w1 = sp.where(x1>=x2.min())[0]
	w2 = sp.where(x2<=x1.max())[0]
	y = sp.arange(data1.shape[0], dtype='i')-247//2
	L = abs(x1.min()-x1.max())
	N = data1.shape[1]
	m = data1.shape[0]
	#x1,x2 = x1*100, x2*100
	l = len(w1)//2
	errorfunction = lambda p: 1/2/N/m*sum((data1[20:-20,w1[l//2:-l//2]]-data2[(20+int(p[0])):(-20+int(p[0])),w2[l:] - int(p[1])])**2)
	res = [[0,0],0]
	try:
		res = optimize.fmin_tnc(errorfunction, approx_grad=1,x0=[-8,len(w1)], bounds=(optimBounds[0], optimBounds[1]), epsilon=eps, fmin=errorfunction((0,0))/10)
	except:
		traceback.print_exc()
		
	print(res)
	p = res[0]
	step = abs(x1[1]-x1[0])
	nx = (l + p[1])/2//step
	print(-step**2*nx)
	X1, Y1, X2, Y2 = x1, y, x2-step**2*nx, y-int(res[0][0])
	'''
	if show:
		mpl.contourf(X1,Y1,data1,50 , cmap=cm.gist_heat_r)
		mpl.contour(X2,Y2,data2,40,cmap=cm.hsv)
		input("next")
	'''
	#if not returnAll:
	return n2, (-nx*step**2-(x1[w1].max() - x1[w1].min()))*N/L, -int(res[0][0]/2), (-nx*step**2-(x1[w1].max() - x1[w1].min()))
	#else:
	#	return X1,Y1,data1, X2,Y2,data2




if __name__ == "__main__":
	
	out = getData(options.dir,  options.background)
	
	import shelve
	'''
	d = shelve.open(os.path.join(options.dir, "res.db"))
	d['res'] = out
	d.close()
	'''

	OUT = {'res': out}
	#del out

	a,b = OUT['res'][0][1].shape

	r = []
	for i in range(len(OUT['res'])-1):
		print (i)
		r.append(test(i,i+1,1,False))

	r = sp.array(r)

	q = [(0,0,0,0)]

	for j,i in enumerate(r):
	    q.append((i[0],r[:(j+1),1].sum(), r[:(j+1),2].sum(), r[:(j+1),3].sum()))
	q = sp.array(q, dtype='i')
	shiftY_min, shiftY_max = q[:,2].min(), q[:,2].max()

	d = sp.zeros((a + shiftY_max*(shiftY_max>0) + shiftY_min*(shiftY_min<0), b*(len(r)+1) + q[:,1][-1]))
	x = sp.linspace(min([OUT['res'][0][0].min(), OUT['res'][-1][0].min()]), max([OUT['res'][0][0].max(), OUT['res'][-1][0].max()])+q[-1,3], d.shape[1])
	y_start = shiftY_min*(shiftY_min<0)
	for j,i in enumerate(q):
		print(j)
		#data2 = OUT['res'][i[0]][1].T[::-1].T
		try:
			data_old = d[(i[2]+y_start):(a+i[2]+y_start), (b*j+i[1]):(b*(j+1)+i[1]) ]
			not_empty = (data_old!=0)
			data_new = OUT['res'][i[0]][1].T[::-1].T
			d[(i[2]+y_start):(a+i[2]+y_start), (b*j+i[1]):(b*(j+1)+i[1]) ] = data_new * (-not_empty) + (data_old + data_new)/2*not_empty

		except:
			traceback.print_exc()
	print(d.shape)

	y = sp.arange(len(d))
	a1 = (y[d[:,0]>0]).mean()
	a2 = (y[d[:,-1]>0]).mean()
	k = (a2-a1)/d.shape[1]

	rotated = ndimage.rotate(d, sp.rad2deg(arctan(k)), reshape=False, order=1)

	a1 = (y[rotated[:,r.shape[1]//2]>0]).mean()
	try:
		rotated = rotated[int(a1 - a/2): int(a1+ a/2),:]
	except:
		traceback.print_exc()
	sp.save(os.path.join(options.dir, "res"), rotated)
	sp.save(os.path.join(options.dir, "res_x"), x)
	#sp.save(os.path.join(options.dir, "res_q"), q)

	#sp.savetxt(os.path.join(options.dir, "res.dat"), d)