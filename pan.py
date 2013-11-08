import sys
import glob
from pylab import *
#from mplwidget import *

fname = glob.glob(sys.argv[1])[0]
r = np.load(fname)

#figure(1,  figsize=(400,2))
ax = subplot(111)


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.matshow(r, cmap=cm.gray)
savefig(fname+'.eps', dpi=1000, format='eps', bbox_inches='tight', pad_inches=0)