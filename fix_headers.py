import pyfits as pf
from glob import glob
import sys

if __name__ == "__main__":

	files = glob(sys.argv[1])
	a = []
	for i in files:
		print(i)
		header = pf.open(i, mode='update')
		bs = header[0].header[5]
		bz = header[0].header[6]
		header[0].header["BSCALE"] = bs
		header[0].header["BZERO"] = bz
		header.flush()
		#try:
		#	c = header[0].header['COMMENT']
		#	a.append([c,i])
			#if c != '':
			#	print(c, i)#bs, bz)
				
			
		#except:
		#	pass
		header.close()
	#for i in a:
	#	print(i)

