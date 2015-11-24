import pyfits as pf
from glob import glob
import sys
from scipy import array, savetxt, save

if __name__ == "__main__":

	files = glob(sys.argv[1])
	a = []
	for i in files:
		#print(i)
		header = pf.open(i, mode='update')
		n = int(header[0].header['FRAMENO'])

		angle = float(header[0].header['ANGLE'])
		coef = float(header[0].header['STEPPERC'])
		a.append([n,angle,coef])
		#bs = header[0].header[5]
		#bz = header[0].header[6]
		#header[0].header["BSCALE"] = bs
		#header[0].header["BZERO"] = bz
		#header.flush()
		#try:
		#	c = header[0].header['COMMENT']
		#	a.append([c,i])
			#if c != '':
			#	print(c, i)#bs, bz)
				
			
		#except:
		#	pass
		header.close()
	a = array(a)
	files = array(files, dtype=str)
	files = files[a[:,0].argsort()]
	old = array([files,a[:,1]], dtype=str) 
	save(sys.argv[1].replace('*/','').replace("*",'0').replace('fits','npy'),old)
	a = a[a[:,0].argsort()]

	r = a[1:,1]-a[:-1,1]
	c = array([r,a[1:,1]]).T
	a0 = a[0,1]
	for j,i in enumerate(files[1:]):
		header = pf.open(i)#, mode='update')
		n = int(header[0].header['FRAMENO'])
		angle_old = float(header[0].header['ANGLE'])
		coef = float(header[0].header['STEPPERC'])
		#comment = header[0].header["COMMENTS"]
		
		#header[0].header.update({"COMMENT": ";OLD_ANGLE=" + str(angle)})
		angle = a0 + int(c[j,0]*126)/126
		header[0].header['ANGLE'] = angle
		a0 = angle
		print('\t\t%f\t\t|\t\t%f\t\t'%(angle_old, angle))
		#header.flush()
	#for i in range(1,len(a[:,1])):
	#	print(a[i,1]-a[i-1:1])
		#r.append(a[i,1]-a[i-1:1])
	print(a)
	print(c)
	#for i in a:
	#	print(i)

