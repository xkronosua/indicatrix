#!/usr/bin/python
# _*_ coding: utf-8 _*_

from optparse import OptionParser
import os
import pyfits as pf
import scipy as sp
import glob
import re
from scipy.signal import medfilt2d
import scipy.interpolate as interp
import json
import traceback
from scipy import ndimage
import matplotlib.pyplot as plt



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
	action='store', default="1064")


parser.add_option("-D", "--distance", dest="DIST", help="set sample-matrix distance, cm",\
	action='store', default="23", type='float')

parser.add_option("-N", dest="N", help="set stripes num",\
	action='store', default="1", type='int')

parser.add_option("-s", dest="slice", help="cut subrect ",\
	action='store', default="")

parser.add_option("-A", '--averaging', dest="average", help="set averaging param",\
	action='store', default="0", type='float')

parser.add_option("-t", "--type", dest="TYPE", help="set files type",\
	action='store', default="fits")

parser.add_option("-j", "--journal", dest="journal", help="set journal file", action='store')

parser.add_option("-d", "--dir", dest="dir", help="set input dir", action='store')
parser.add_option("-o", "--out", dest="outFile", help="set output file", action='store', default="res.dat")

parser.add_option("-b", "--background", dest="background", help="set background for all files", action='store', default='')
parser.add_option("--bInterp", dest="bInterp", help="interpolate background`s", action='store_true')
parser.add_option("--bDict", dest="bDict", help="create background`s dict[angle]", action='store_true')

parser.add_option("--bFilt", dest="bFilt", help="set background`s filter ", action='store', default='')
parser.add_option("-z", "--zero", dest="zero", help="move start", action="store_true")
parser.add_option("-p", "--plot", dest="plot", help="plot data", action="store_true")
parser.add_option("-P", "--panorama", dest="panorama", help="create panorama", action="store_true")
parser.add_option("-m", "--medfilt", dest="medfilt", help="use median filter", default=0, type='int', action="store")
parser.add_option("-g", "--gaussfilt", dest="gaussfilt", help="use gauss filter", default='1x1', action="store")
parser.add_option("--bounds", dest="bounds", help="cut min and max", default='x', action="store")
parser.add_option("--power", dest="power", help="P_out", default=-1., type='float', action="store")
parser.add_option("-e", "--exposure", dest="exposure", help="exposure value. -1 for auto", default=-1, type='float', action="store")

(options, args) = parser.parse_args()

print( options, args)
#################################################################
# Знайдемо кут, що захоплює матриця
MATRIX_SIZE = [0.36, 0.49]#[494.*7.4*10**-4, 659.*7.4*10**-4]	# см
theta = sp.degrees( sp.arctan(MATRIX_SIZE[1]/options.DIST))/2.
print( "theta: %f"%theta)
N = options.N	# кількість смуг
theta_range = None
if N == 1:
	theta_range = sp.array([0])	
else:
	theta_range = sp.linspace(-theta, theta, N)

#print theta_range
def cut_minmax(data, bounds):
	if len(bounds) != 0:
		
		data *= (data>bounds[0])*(data<bounds[1])
	return data

def filtCalc(filters, filtTable=None):
	if filters:
		if not filtTable is None:
			filters = filters.replace(' ', '').replace(' ', '').replace('+', ',').replace(';', ',').replace('.', ',')
			res = 1.
			try:
				res = sp.multiply.reduce( [ filtTable[options.LENGTH][i.upper()] for i in filters.split(",")] ) 
			except KeyError:
				traceback.print_exc()
				res = 1.
			return res
		else:
			return	
	else:
		return 1.


def getData(Dir, theta_range, bgfile=''):
	sh = []
	out = []

	try:
		bounds = [int(i) for i in options.bounds.split('x')]
	except ValueError:
		bounds = []



	if os.path.exists(Dir):
		filtTable = json.load(open(options.FILTERS))
		
		# background
		bgData = 1.
		bDict = {}
		bOut = []
		data = []
		if bgfile:
			if options.bInterp:
				backgrounds = glob.glob(bgfile)
				

				for i in backgrounds:
					background = pf.open(i)[0]
					data = background.data
					print( sp.shape(data))
					if 'FILTER' in  background.header.keys():
						bgFilt = filtCalc(background.header['FILTER'], filtTable)
					else:
						bgFilt = 1.
					data = data / bgFilt
					if options.exposure != -1:
						data /= float(options.exposure)*1000
					else :
						data /= float(background.header['EXPTIME'])*1000
			
					if options.power != -1:
						data /= float(options.power)
					else :
						data /= float(background.header['POWER'])
					bname = background.header['OBJECT']
					print(bname, i)
					try:
						angle = float(background.header['ANGLE'])
						print("angle : {:.3f}".format(angle))
						bOut.append([angle, data.mean()])
					except:
						traceback.print_exc()
				bOut = sp.array(bOut)
				bOut = bOut[bOut[:,0].argsort()]
				#bOut[:,0] = bOut[:,0][::-1]
				plt.plot(bOut[:,0],bOut[:,1])
				plt.show(False)
				bgData = sp.poly1d(sp.polyfit(bOut[:,0],bOut[:,1], 5))
			elif options.bDict:
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
						print(background.header.keys())
						bgFilt = 1.
					data = data / bgFilt
					bname = background.header['OBJECT']
					print(bname, bgFilt, i)
					try:
						angle = float(background.header['ANGLE'])
						print("angle : {:.3f}".format(angle))
						if options.zero:
							angle -= (angle>180)*360
						bOut.append([angle, data.mean()])
						bDict[angle] = data
					except:
						traceback.print_exc()
						
				bOut = sp.array(bOut)
				bOut = bOut[bOut[:,0].argsort()]
			else:
				background = pf.open(glob.glob(bgfile)[0])[0]
				print( background.data	)
				data = pf.getdata(glob.glob(bgfile)[0])#background.data
				
				print( sp.shape(data))
				if 'FILTER' in  background.header.keys():
					bgFilt = filtCalc(background.header['FILTER'], filtTable)
				else:
					bgFilt = 1.
				bgData = data / bgFilt
				
				if options.exposure != -1:
					bgData /= float(options.exposure)*1000
				else :
					bgData /= float(background.header['EXPTIME'])*1000
			
				if options.power != -1:
					bgData /= float(options.power)
				else :
					bgData /= float(background.header['POWER'])
				
		else:
			print('No BG')
			pass
		sh = sp.shape(data)
		#if options.medfilt:
		#	bgData = medfilt2d(bgData)
		#print(bgData)

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
					print ("\033[1;41m", prof[-1].header['FILTER'], "\033[1;m")
					profFilt = filtCalc(prof[-1].header['FILTER'], filtTable)
				else:
					profFilt = 1.
				profName = prof[-1].header['OBJECT']
				try:
					angle = float(prof[-1].header['ANGLE'])
				except:
					traceback.print_exc()
					print (f)
					continue
				print(f)
				print("angle : {:.3f}".format(angle))
				profData = cut_minmax(profData, bounds)


				profData = profData / profFilt
				if options.exposure != -1:
					profData /= float(options.exposure)*1000
				else :
					profData /= float(prof[-1].header['EXPTIME'])*1000
				
				if options.power != -1:
					profData /= float(options.power)
				else :
					profData /= float(prof[-1].header['POWER'])
					
			except (IOError, ValueError, IndexError):
				traceback.print_exc()
				continue

			'''
			try:
				degrees, minutes = profName.split('_')
				angle = float(degrees) + float(minutes) / 60.
			except ValueError:
				traceback.print_exc()
				continue
			'''
			
			signal = []
			if options.bInterp:
				try:
					
					signal = profData - bgData(angle)
					print(signal.min())
					print("="*20)
				except:
					traceback.print_exc()
					print(angle)
					print(sp.mean(bOut[:,1]), type(bOut))
					print( angle, bOut[:,0])
					try:
						signal = profData - sp.mean(bOut[:,1])
					except:
						traceback.print_exc()
					print(signal.min())
			elif options.bDict:
				try:
					
					signal = abs(profData - bDict[angle])
					print(signal.min())
					print("="*20)
				except:
					traceback.print_exc()
					print(angle)
					
					try:
						signal = profData - sp.mean(bOut[:,1])
					except:
						traceback.print_exc()
					print(signal.min())
			else:
				signal = profData - bgData.mean()

			#signal = signal * ( ( signal - signal.min() ) > ( signal.max()-signal.min() )/100 )
			signal /= options.power

			gf = [ int(i) for i in options.gaussfilt.split('x')]
			signal = ndimage.gaussian_filter(signal, gf)

			if options.medfilt:
				signal = medfilt2d(signal, options.medfilt)
			
			if options.panorama:
				theta_range = sp.linspace(-theta, theta, sp.shape(signal)[1])
			t = angle + theta_range[::-1]
			
			if options.slice:
				try:
					size = [ int(i) for i in options.slice.split('x')]
					#s = sp.shape(size)
					signal = signal[size[0]:-size[0], size[1]:-size[1]]
					t = t[size[1]:-size[1]]
					sh = (sh[0]-2*size[0], sh[1]-2*size[1])
				except:
					traceback.print_exc()
			
			#print(sh, signal.shape)
			
			#if options.panorama and sh == sp.shape(signal):
				#N = sp.shape(signal)[1]
				#stripes = sp.array_split(signal, N, axis=1)
				#for i in range(len(stripes)):
				#	out.append([t[i]]+stripes[::-1][i].T.tolist()[0])
			
			tt = sp.vstack((t[::-1],signal)).T
			print(sp.shape(tt),)
				
			out.append(tt)
					#sh.append(sp.shape(tt))
			#	else:
			#		print('?????')
			'''
				stripes = sp.array_split(signal, N, axis=1)
				print sp.shape(stripes[0]), sp.shape(signal)
				res = [ a.mean() for a in stripes[::-1]]
				print len(res)
				for i in range(len(res)):
					out.append([t[i], res[i]])
				''' 
			'''
			# Конвертація в gif
			if options.plot:
				imgDir = os.path.join(Dir,"img")
				if not os.path.exists(imgDir):
					os.mkdir(imgDir)
				tmpName = os.path.join(Dir,"img", str(degrees) + "_" + str(minutes) + ".dat")
				print tmpName, sp.shape(signal)
				#print ">>>>", signal
				sp.savetxt(tmpName, signal)
				os.system('sh ./plotmap.sh ' + tmpName)
				os.remove(tmpName)
			'''
	print(sh)	
	if options.panorama :
		try:
			out = sp.vstack(out)
		except:
			traceback.print_exc()
	else:
		out = sp.array(out)
	#out = out.astype('int64')
	if len(out)>=1 and not options.zero: out[:,0] = out[:,0]+(out[:,0]<180)*360
	#out[:,1] = out[:,1] + 10**9
	#print errList,"\nLen:\n\tjournal = " + str(len(journal)) + "\tout = " + str(len(out))
	#if options.average and len(theta_range)>1:
	#	out = average(out, options.average*(theta_range[1] - theta_range[0]))
	#if options.panorama:

	print( theta_range[-1]-theta_range[0])
	print(sh)
	step = theta_range[1]-theta_range[0]
	return out, step

	

tmp=[]

if __name__ == "__main__":
	#(options,args) = parser.parse_args()
	'''
	N = options.N	# кількість смуг
	theta_range = None
	if N == 1:
		theta_range = sp.array([0])	
	else:
		theta_range = sp.linspace(-theta, theta, N)
	'''
	A, step = getData(options.dir, theta_range, options.background)
	print(sp.shape(A))
	print( theta_range[-1]-theta_range[0])
	#print theta_range
	#step = theta_range[1] - theta_range[0]
	#sp.savetxt(os.path.join(options.dir,options.outFile),A)
	if options.panorama:
		A = A[A[:,0].argsort()]
		A = A[sp.unique(A[:,0],return_index=True)[1]]
		x=A[:,0]
		
		#A[:,1]+=10**8
		
		z=A[:,10:-10]#.astype('int32')
		s=sp.shape(z)
		print (s)
		x2=sp.arange(x.min(),x.max(),step*3)#0.01/3.3 )
		y=sp.linspace(0,s[1],s[1])

		import scipy.interpolate
		w = scipy.where(z==z.max())[1][0]
		print (scipy.shape(z[:,87]), scipy.shape(x), w)
		#Z_new = scipy.interpolate.interp1d(x,z[:,w])(x2)
		#z_new2D = []
		
		#print sp.shape(z)
		#while i<x.max():
		#	w=(x>=i)*(x<(i+step*3))
		#	if w.sum() >1:
		#		z_new2D.append( z[w].mean(axis=0))
		#	else: z_new2D.append(z[w])
		#Z_new2D = sp.array(z_new2D)
		'''
		xmax = x.max()
		ymax = y.max()
		zmax = z.max()
		x /= xmax
		y /= ymax
		z /= zmax
		try:
			
			Z_new2D = scipy.interpolate.RectBivariateSpline(x, y, z, kx=5, ky=5, s=0)(x2/xmax,y/ymax)
		except:
			traceback.print_exc()
			Z_new2D = z
		Z_new2D *= zmax
		x *= xmax
		y *= ymax 
		z *= zmax
		#matshow(Z_new)
		#matshow(Z_new.T)
		#matshow(Z_new.T, cmap = cm.Greys_r)

		out = scipy.array([x,z[:,w]]).T
		print (scipy.shape(out))
		'''
		sp.save(os.path.join(options.dir,options.outFile),scipy.array([x, z[:,w]]).T)
		#sp.save(os.path.join(options.dir,options.outFile),scipy.array([x, ndimage.gaussian_filter( z,(7,7))[:,w]]).T)#Z_new.T.round(0))
		'''
		try:
			sp.save(os.path.join(options.dir,options.outFile+'2D'),Z_new2D.T.round(0))
			sp.save(os.path.join(options.dir,options.outFile+'x'),x)
			sp.save(os.path.join(options.dir,options.outFile+'y'),y)
			sp.save(os.path.join(options.dir,options.outFile+'z'),z)

			tmp = z
		except:
			pass
		'''
		
		'''
		Z_new=abs(Z_new)+1.
		z_new=sp.log10(Z_new)
		print (Z_new.min(),Z_new.max())
		import scipy.misc
		img = scipy.misc.toimage(Z_new, high=250, low=0, mode='I')
		#s = sp.shape(img)
		#img = scipy.misc.imresize(img, size=(s[0]*2,s[1]*2))
		scipy.misc.imsave(os.path.join(options.dir,options.outFile)+'.tiff', img)
		'''

'''
def F(name, r=''):
    out = []
    print glob.glob(name+'/*.fits')
    h=pf.open(glob.glob(name+'/*.fits')[0], mode='update')
    if r=='':
        out=(name, h[0].header['FILTER'])
    else:
        h[0].header.update('FILTER',r)
        h.flush()
        out=(name, h[0].header['FILTER'])
    h.close()
    return out
'''
