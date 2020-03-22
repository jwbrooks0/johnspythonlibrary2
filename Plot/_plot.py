
###################################################################################
### import libraries
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import matplotlib as _mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################################
### figure/ax related

def subTitle(ax,string,
			xy=(0.5, .98),
			box=True,
			textColor='k',
			xycoords='axes fraction',
			fontSize=8,
			horizontalalignment='center',
			verticalalignment='top'):
	"""
	wrapper for the annotate axis function.  the default setting is for a
	figure subtitle at the top of a particular axis
	
	Parameters
	----------
	ax : matplotlib.axes._subplots.AxesSubplot
		Axis that will receive the text box
	string : str
		String to put in textbox
	xy : tuple
		(x,y) coordinates for the text box
	box : bool
		True - Creates a box around the text
		False - No box
	textColor : str
		text color
	xycoords : str
		type of coordinates.  default = 'axes fraction'
	fontSize : int
		text font size
	horizontalalignment : str
		'center' - coordinates are cenetered at the center of the box
		'left'
		'right'
	verticalalignment : str
		'top' - coordinates are centered at the top of the box
	
	TODO(John) Expand functionality
	"""
	if box==True:
		box=dict(boxstyle="square, pad=.25", fc="w",edgecolor='k')
	else:
		box=None

	ax.annotate(string, 
				xy=xy, 
				color=textColor,
				xycoords=xycoords, 
				fontsize=fontSize,
				horizontalalignment=horizontalalignment, 
				verticalalignment=verticalalignment,
				bbox=box)



def finalizeFigure(fig,
				   title='',
				   h_pad=0.25,
				   w_pad=0.25, 
				   fontSizeTitle=12,
				   figSize=[],
				   pad=0.5):
	""" 
	Performs many of the "same old" commands that need to be performed for
	each figure but wraps it up into one function
	
	Parameters
	----------
	fig : matplotlib.figure.Figure
		Figure to be modified
	title : str
		(Optional) Figure title
	"""
	
	# fig.suptitle(title) # note: suptitle is not compatible with set_tight_layout
	
	if title!='':
		fig.axes[0].set_title(title,fontsize=fontSizeTitle)
		
	if figSize!=[]:
		fig.set_size_inches(figSize)
		
#	fig.set_tight_layout(True)
	fig.tight_layout(h_pad=h_pad,w_pad=w_pad,pad=pad) # sets tight_layout and sets the padding between subplots
			
	
	
def legendOutside(	ax,
				    ncol=1,
					xy=(1.04,1),
					refCorner='upper left',
					fontSizeStandard=8,
					handlelength=2,
					numberLegendPoints=2,
					labelspacing=0.1):
	"""
	Places a legend outside (or anywhere really) in reference to an axis object
	
	References
	----------
	https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
	"""
	x,y=xy
	ax.legend(	bbox_to_anchor=(x,y),
				loc=refCorner,
				fontsize=fontSizeStandard,
				numpoints=numberLegendPoints,
				ncol=ncol,
				handlelength=2,
				labelspacing=labelspacing)
			
	

def positionPlot(	fig,
					size=[],
					left=None, 
					bottom=None, 
					right=None, 
					top=None, 
					wspace=None, 
					hspace=None):
	"""
	left  = 0.125  # the left side of the subplots of the figure
	right = 0.9    # the right side of the subplots of the figure
	bottom = 0.1   # the bottom of the subplots of the figure
	top = 0.9      # the top of the subplots of the figure
	wspace = 0.2   # the amount of width reserved for blank space between subplots
	hspace = 0.2   # the amount of height reserved for white space between subplots
	
	Reference
	---------
	https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	"""
	if size!=[]:
		fig.set_size_inches(size)
	fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	
	

def finalizeSubplot(	ax,
						xlabel='',
						ylabel='',
						title='',
						subtitle='',
						xlim=[],
						ylim=[],
						fontSizeStandard=8, 
						fontSizeTitle=8,
						legendLoc='best', 
						color='grey',
						linestyle=':',
						alpha=1.0,
						yticks=[],
						ytickLabels=[],
						xticks=[],
						xtickLabels=[],
						yAxisColor='k',
						legendOn=True,
						ncol=1,
						labelspacing=0.1,
						numberLegendPoints=2,
						handlelength=2,
						tickDirection='out',
						):
	"""
	Performs many of the "same old" commands that need to be performed for
	each subplot but wraps it up into one function
	
	Parameters
	----------
	ax : matplotlib.axes._subplots.AxesSubplot
		figure axis to be modified
	xlabel : str
		x label
	ylabel : str
		y label
	title : str
		title
	xlim : tuple or list of two floats
		x limits of plot
	ylim : tuple or list of two floats
		y limits of plot
	fontSizeStandard : int
		font size of everything but the title
	fontSizeTitle : int
		font size of title
	legendLoc : str
		location of legend
	color : str
		color for axis markings
	linestyle : str
		linestyle for axis markings
	alpha : float
		value between 0 and 1 for axis markings
	
	# TODO(John) Add font sizes, axis ticks, and axis tick labels
	"""
#	legend=False
	
	# check to see if ax is a list or array
	if type(ax)==list or type(ax)==_np.ndarray:
		pass
	else:
		ax=[ax]
		
	# allows it to handle a list of axes
	for i in range(0,len(ax)):
		
		# title and axis labels
		ax[i].set_ylabel(ylabel,fontsize=fontSizeStandard,color=yAxisColor)
		if i==0:
			ax[i].set_title(title,fontsize=fontSizeTitle)
		if i==len(ax)-1:
			ax[i].set_xlabel(xlabel,fontsize=fontSizeStandard)
		
		# subtitle
		if subtitle!='':
			subTitle(ax[i],subtitle,fontSize=fontSizeStandard)
		
		# add a legend only if "any" plot label has been defined # TODO(John) does not work for multiple columns of subplots.  Fix
		if legendOn==True:
#			for j in range(len(ax[i].lines)):
#				label=ax[i].lines[j].get_label()
##				if label[0]==u'_':
##					legendOn=False
			if legendOn==True:
				ax[i].legend(	fontsize=fontSizeStandard,
							loc=legendLoc,
							numpoints=numberLegendPoints, # numpoints is the number of markers in the legend
							ncol=ncol,
							labelspacing=labelspacing,
							handlelength=handlelength) 
				
		# set x and y axis tick label fontsize
		ax[i].tick_params(	axis='both',
						labelsize=fontSizeStandard,
						direction=tickDirection
						)
			
		# x and y limits
		if len(xlim)>0:
			ax[i].set_xlim(xlim)
		if len(ylim)>0:
			ax[i].set_ylim(ylim)
			
		# y ticks and y tick labels
		if yticks!=[]:
			ax[i].set_yticks(yticks)
		if ytickLabels!=[]:
			ax[i].set_yticklabels(ytickLabels)
			
		# y ticks and y tick labels
		if xticks!=[]:
			ax[i].set_xticks(xticks)
		if xtickLabels!=[]:
			ax[i].set_xticklabels(xtickLabels)
			
		# 
		ax[i].tick_params(axis='y', labelcolor=yAxisColor)
			
		# add dotted lines along the x=0 and y=0 lines
		ylim=ax[i].get_ylim()
		xlim=ax[i].get_xlim()
		ax[i].plot(xlim,[0,0],color=color,linestyle=linestyle,alpha=alpha)
		ax[i].plot([0,0],ylim,color=color,linestyle=linestyle,alpha=alpha)
		ax[i].set_ylim(ylim)
		ax[i].set_xlim(xlim)
		
	
		
		
def subplotsWithColormaps(nrows=2,sharex=False):
	"""
	Initialize subplots if any plot requires a colorbar.
	
	Parameters
	----------
	nrows : int
		number of rows of subplots
	sharex : bool
		sharex, same as with plt.subplots()
		
	Returns
	-------
	
	Example
	-------
	::
		
		fig,ax,cax=subplotsWithColormaps(3,sharex=True)
		
		t=np.arange(0,10e-3,2e-6)
		y=np.sin(2*np.pi*2e3*t)
		c=np.sin(2*np.pi*2e3*t)
			
		im1 = ax[0].scatter(t, y, c=c, cmap='magma')
		fig.colorbar(im1, ax=ax[0],cax=cax[0])
		ax[1].plot(t,y)
		ax[2].plot(t,y)
		cax[1].remove()
		cax[2].remove()
	"""
	fig, ax = _plt.subplots(nrows,1,sharex=sharex)
	
	if nrows>1:
		divider=[]
		cax=[]
		for i in range(len(ax)):
			divider.append(make_axes_locatable(ax[i]))
			cax.append(divider[i].append_axes("right", size="2%", pad=.05))
	else:
		divider=make_axes_locatable(ax)
		cax=divider.append_axes("right", size="2%", pad=.05)
			
	return fig,ax,cax



###################################################################################
### shapes
def arrow(ax,xyTail=(0,0),xyHead=(1,1),color='r',width=3,headwidth=10,headlength=10,alpha=1.0):
	"""
	Draws an arrow
	
	References
	----------
	https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.annotate.html
	"""
	ax.annotate("", xy=xyHead, xytext=xyTail, arrowprops=dict(	width=width,
														headwidth=headwidth,
														headlength=headlength,
														color=color,
														alpha=alpha),
				label='current')
	

def rectangle(ax,x_bottomleft,y_bottomleft,width,height,color='r',alpha=0.5):
	"""
	Adds a colored rectangle to a plot
	
	ax : matplotlib.axes._subplots.AxesSubplot
		axis handle for the rectangle
	x_bottomleft : float
		x-coordinate of the bottom left corner of the rectangle
	y_bottomleft : float
		y-coordinate of the bottom left corner of the rectangle
	width : float
		width of rectangle
	height : float
		height of the rectangle
	color : str
		color of the rectangle
	alpha : float
		alpha (transparency) value of the rectangle, between 0.0 and 1.0.  
		
	Example
	-------
	::
		
		import numpy
		import matplotlib.pyplot as plt
		fig,ax=plt.subplots()
		ax.plot(np.arange(0,10))
		rectangle(ax,1,1,7,2.5)
	"""
		

	from matplotlib.patches import Rectangle
	from matplotlib.collections import PatchCollection

	rect=Rectangle((x_bottomleft,y_bottomleft),width,height)
	pc=PatchCollection([rect],facecolor=color,alpha=alpha)
	ax.add_collection(pc)




###################################################################################
### specialty plots
def contourPlot(	ax,
					x,
					y,
					z,
					levels,
					ylabel='',
					zlabel='',
					yticklabels=None,
					xlabel='',
					title='',
					zlim=[],
					zticklabels='',
					ztickLabels=[],
					colorMap=_plt.cm.viridis,
					fill=True,
					fontsize=8,
					colorBar=True):
	"""
	wrapper for contour plot.  under development
	#TODO this needs an overhaul
	"""
	X,Y=_np.meshgrid(x,y)
	if len(zlim)>0:
		vmin=zlim[0]
		vmax=zlim[1]
	else:
		vmin=None
		vmax=None
#	if levels!=[]:
	if fill==True:
		CS=ax.contourf(X,Y,z,levels=levels,cmap=colorMap,vmin=vmin,vmax=vmax)#vmin=zlim[0],vmax=zlim[1],
	else:
		CS=ax.contour(X,Y,z,levels=levels,cmap=colorMap,vmin=vmin,vmax=vmax)#vmin=zlim[0],vmax=zlim[1],

	ax.set_xlabel(xlabel,fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	ax.set_title(title,fontsize=fontsize)
	if type(yticklabels) != type(None):
		ax.set_yticks(y)
		ax.set_yticklabels(yticklabels)
	if colorBar==True:
		if zticklabels!='':
			cbar = _plt.colorbar(CS,ax=ax,ticks=zticklabels,pad=0.01)
			cbar.ax.set_yticklabels(ztickLabels,fontsize=fontsize)
		else:
			cbar = _plt.colorbar(CS,ax=ax,pad=0.01)
		cbar.ax.set_ylabel(zlabel,fontsize=fontsize)
	ax.tick_params(axis='x',labelsize=fontsize)
	ax.tick_params(axis='y',labelsize=fontsize)
	
	
	for c in CS.collections:
	    c.set_edgecolor("face")
		