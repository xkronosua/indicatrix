#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from optparse import OptionParser
import os
import scipy as sp


usage = """usage: %prog [options] file
press 'x' key with :
	concatenate - mouseButton1 (Left button)
	undo - mouseButton3 (Right button)
"""
parser = OptionParser(usage=usage)
parser.add_option("-p", "--part", action="store", dest="part", default="l", help="part of indicatrix (left, right). default=[default]")
parser.add_option("-s", "--subplot", action="store_true", dest="subplot", default=False, help="plot in two windows: real + log. default=[default]")


class logConcat:
	x ,y, history = [], [], []
	subplot = False
	part = ""
	filename = './out'
	def __init__(self, fname = './out', data = [], Part = 'l', subplot = False ):
		self.subplot = subplot
		self.part = Part
		self.filename = fname
		if fname:	
			data = sp.loadtxt(fname)
		else: pass
		self.x = data[:,0]
		self.update(Y = data[:,1])

		self.logConcat()
		
	def update(self, Y = [], action = 1):
		if action == 1:
			self.y = Y
			self.history.append(Y)
		elif action == -1 and len(self.history) > 1:
			self.history.pop()
			self.y = self.history[-1]
		else: print(len(self.history))
		
		return self.y
	def Save(self,event):
		f = self.filename
		#ex = f.split('.')[-1]
		#sp.savetxt(f.split('.'+ex)[0])
		sp.savetxt(f+"concat.dat",sp.array([self.x,self.y]).T)
		print("Save to" + f+"concat.dat")
	
	def logConcat(self):
		
		fig = plt.figure(1)
		if self.subplot:
			self.ax0 = fig.add_subplot(212)
			self.ax = fig.add_subplot(211)
			self.Plot(self.x, self.y)
		else:
			self.ax = fig.add_subplot(111)
			self.Plot(self.x, self.y)
		save = plt.axes([0.90, 0.90, 0.1, 0.075])
		self.save_button = Button(save, 'Save')
		self.save_button.on_clicked(self.Save)	
		cid = fig.canvas.mpl_connect('button_press_event',self.onclick)
		#cid1 = fig.canvas.mpl_connect('key_release_event',self.Save)
		
		plt.show()
	def Plot(self, x, y, style ="go"):
		if self.subplot:
			if len(self.history) > 1:
				self.ax.cla()
				self.ax0.cla()
			else: pass
			self.ax0.plot(x, y, 'go')	
			self.ax0.grid(True)
			self.ax.semilogy(x, y, style)	
			self.ax.grid(True)
		else: 
			if len(self.history) > 1:
				self.ax.cla()
			else: pass
			self.ax.semilogy(x, y, style)	
			self.ax.grid(True)
	
	def onclick(self, event):
		x = self.x
		y = self.y		
		if event.button==3 and event.key == 'x':
			#plt.hold(False)
			y = self.update(action = -1)
			'''
			if self.subplot and len(self.history) > 1:
				self.ax.cla()
				self.ax0.cla()
			elif not self.subplot and len(self.history) > 1:
				self.ax.cla()
			'''
			self.Plot(x, self.y, style = 'bo')

			plt.draw()
			
		elif event.button==1 and event.key == 'x':
			'''
			if self.subplot and len(self.history) > 1:
				self.ax.cla()
				self.ax0.cla()
			elif not self.subplot and len(self.history) > 1:
				self.ax.cla()
			'''
			#self.Plot(x, self.y, style = 'bo')

			diff = abs(x-event.xdata)
			c_from = sp.where(diff==diff.min())[0]
			print c_from
			y_new = []
			if self.part == "l":
				y_new = sp.concatenate([y[:c_from],y[c_from:]*y[c_from-1]/y[c_from] ])
			elif self.part == "r":
				y_new = sp.concatenate([y[:c_from]*y[c_from]/y[c_from-1],y[c_from:] ])
			else: print("key error")
			#print (x-event.xdata)**2+(y-event.ydata)**2
			#print x_del, y_del
			self.ax.semilogy(x[c_from],y[c_from],'rx',markersize=20)
			self.Plot(x, y_new, style = 'go')
			self.update(Y=y_new)
				
			plt.hold(False)
			plt.draw()
			print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
			
				
				
		
	
if __name__ == "__main__":
	(options, args) = parser.parse_args()
	name = args[0]
	d = logConcat(fname = name, Part = options.part, subplot = options.subplot)

