# IPython log file
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import pyfits as pf

data=pf.getdata('8_0_0001.fits')
m,n=247,330
x=arange(n)
y=arange(m)

X,Y=meshgrid(x,y)
from scipy import ndimage
x_c=(data*X).sum()/data.sum()
y_c=(data*Y).sum()/data.sum()
D4s_y=4*sqrt((data*(Y-y_c)**2).sum()/data.sum())
D4s_x=4*sqrt((data*(X-x_c)**2).sum()/data.sum())

Z=ndimage.gaussian_filter(data/data.max(),(1,1))
rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d',  axisbg='#ffffff')
ax.set_ylabel("Y, px")
ax.set_xlabel("X, px")
ax.set_zlabel("Intensity, arb. un.")
ax.text(-200,150,1,r"$\sigma_x = %.2f,\; \sigma_y = %.2f,\; x_c = %.2f,\; y_c = %.2f$" % (D4s_x/4, D4s_y/4, x_c, y_c))
surf = ax.plot_surface(X, Y, Z, cmap=cm.bone, rstride=5, cstride=5, linewidth=0.05, antialiased=True)
draw()
show()