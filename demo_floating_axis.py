'''
"""
An experimental support for curvilinear grid.
"""


def curvelinear_test2(fig):
    """
    polar projection, but in a rectangular box.
    """
    global ax1
    import numpy as np
    import  mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D

    from mpl_toolkits.axisartist import SubplotHost

    from mpl_toolkits.axisartist import GridHelperCurveLinear

    # see demo_curvelinear_grid.py for details
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 180,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)

    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    # Now creates floating axis

    #grid_helper = ax1.get_grid_helper()
    # floating axis whose first coordinate (theta) is fixed at 60
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 0)
    #axis.label.set_text(r"$\theta = 60^{\circ}$")
    axis.label.set_visible(True)

    # floating axis whose second coordinate (r) is fixed at 6
    ax1.axis["lon"] = axis = ax1.new_floating_axis(0, 0)
    #axis.label.set_text(r"$r = 6$")
    #axis.set_rscale('log')
    ax1.set_aspect(1.)
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(0, 13)
    #ax1.set_yscale('log')
    ax1.grid(True)

    a1=np.loadtxt('data/Jan2415/s1/s1.dat')
    a3=np.loadtxt('data/Jan2415/s3/s3.dat')
    ax1.plot(a1[:,0], np.log10(a1[:,1]), 'r')
    ax1.plot(a3[:,0], np.log10(a3[:,1]), 'g')

import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(5, 5))
fig.clf()

curvelinear_test2(fig)

plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

bazbins = np.linspace(0, 2*np.pi, 360)
fbins = np.logspace(np.log10(0.05), np.log10(0.5), 101)
theta, r = np.meshgrid(bazbins, fbins) 

# Set up plot window
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(projection='polar'))

# polar
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
ax.set_rscale('log')

a1=np.loadtxt('data/Jan2415/s1/s1.dat')
a3=np.loadtxt('data/Jan2415/s3/s3.dat')
#plt.gca().invert_yaxis()

ax.plot(np.deg2rad(a1[:,0]+180), a1[:,1], '.r', markersize=0.9)
ax.plot(np.deg2rad(a3[:,0]+180), a3[:,1], '.g', markersize=0.9)



ax.set_ylim((0.6, max(a1[:,1].max(), a3[:,1].max())*2))
ax.set_rgrids([10**2, 10**5, 10**8, 10**11,  a1[:,1].max()*10], angle=145.)
plt.savefig("res.png", format="png", dpi=300)
plt.show()
