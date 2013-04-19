#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import glob
import sys

argv = sys.argv[1:]
imgNames = []
for i in argv:
	Dir = i
	if Dir[-1] == '/':
		Dir = Dir[:-1]
	imgNames += glob.glob(Dir + '/*.gif')
imgNames.sort()

print imgNames
# First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.subplot(111)
#imgplot, = ax.imshow(os.path.join())

fig = plt.figure()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in imgNames:

    data = plt.imread(i)
    im = plt.imshow(data)
    
    ims.append([im])
interv = 200
ani = animation.ArtistAnimation(fig, ims, interval=interv, blit=True,
    )

ani.save('dynamic_images.mp4')


plt.show()
