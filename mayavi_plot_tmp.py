from numpy import *
from mayavi import mlab
import pyfits as pf
from scipy import ndimage
import glob
X,Y = meshgrid(arange(330),arange(247))

data1 = pf.getdata(glob.glob("147*/*.fits")[0])
data2 = pf.getdata(glob.glob("148*/*.fits")[0])

Z1 = ndimage.gaussian_filter(data1/300,(1,1))
Z2 = ndimage.gaussian_filter(data2/300,(1,1))

mlab.mesh(X,Y,Z1, colormap="gist_yarg")
mlab.mesh(X-262,Y,Z2, colormap="gray")
mlab.mesh(X[:,-70:]-262,Y[:,-70:],Z2[:,-70:]+10, colormap="gray")


mlab.show()