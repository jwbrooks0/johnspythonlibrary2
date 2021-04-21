
###################################################################################
### import libraries
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import matplotlib as _mpl	
import matplotlib.animation as _animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


###############################################################################
#%% custom matplotlib settings

# custom color cycle.  now 15 colors long
_colors=[ 	"k",
			"tab:blue",
			"tab:orange",
			"tab:green",
#			"tab:red",
			"tab:purple",
			"tab:brown",
			"tab:pink",
			"tab:gray",
			"tab:olive",
			"tab:cyan",
			"lime",
			"blue",
			"red",
			"m"]
_mpl.rcParams['axes.prop_cycle'] = _mpl.cycler(color=_colors)

# custom linewidth
_mpl.rcParams['lines.linewidth'] = 1.0

# custom fontsizes
_FONTSIZE=8
_mpl.rcParams['xtick.labelsize'] = _FONTSIZE
_mpl.rcParams['ytick.labelsize'] = _FONTSIZE
_mpl.rcParams['font.size'] = _FONTSIZE
_mpl.rcParams['legend.fontsize'] = _FONTSIZE
_mpl.rcParams['axes.titlesize'] = _FONTSIZE+2

# legend
_mpl.rcParams['legend.numpoints'] = 3  # not working

# other
_mpl.rcParams.update({'figure.autolayout': True})

###################################################################################
#%% figure/ax related

def subTitle(	ax,
				string,
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
				   figSize=[],
				   **kwargs):
	""" 
	Performs many of the "same old" commands that need to be performed for
	each figure but wraps it up into one function
	
	Parameters
	----------
	fig : matplotlib.figure.Figure
		Figure to be modified
	figSize : list of floats
		Figure size.  Units in inches.  E.g. figSize=[6,4]
	"""
	
	# default input parameters
	params={	'title':'',
				'h_pad' : 0.25,
				'w_pad' : 0.25,
				'fontSizeTitle':_FONTSIZE+2,
				'pad':0.5,
				}
	
	# update default input parameters with kwargs
	params.update(kwargs)
	
	# fig.suptitle(title) # note: suptitle is not compatible with set_tight_layout
	
	if params['title']!='':
#		fig.axes[0].set_title(params['title'],fontsize=params['fontSizeTitle'])
		fig.suptitle(params['title'],fontsize=params['fontSizeTitle'])
		
	if figSize!=[]:
		fig.set_size_inches(figSize)
		
#	fig.set_tight_layout(True)
	fig.tight_layout(h_pad=params['h_pad'],w_pad=params['w_pad'],pad=params['pad']) # sets tight_layout and sets the padding between subplots
			
	
	
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
	right = 0.9	# the right side of the subplots of the figure
	bottom = 0.1   # the bottom of the subplots of the figure
	top = 0.9	  # the top of the subplots of the figure
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
						**kwargs
						):
	"""
	Performs many of the "same old" commands that need to be performed for
	each subplot but wraps it up into one function
	
	Parameters
	----------
	ax : matplotlib.axes._subplots.AxesSubplot
		figure axis to be modified
	kwargs : dict
		misc. parameters for the plot

	Examples
	--------
	Example1::
		
		t=_np.arange(0,10e-3,2e-6)
		y1=_np.cos(_np.pi*2*t*2e3)
		y2=_np.sin(_np.pi*2*t*2e3)
		y3=_np.sin(_np.pi*2*t*3e3)
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(t*1e3,y1,label='y1')
		ax[1].plot(t*1e3,y2,label='y2')
		ax[1].plot(t*1e3,y3,label='y3')
		finalizeSubplot( 	ax[0],
							ylabel='y',
							subtitle='fig1',
							title='title!',
							)
		finalizeSubplot( 	ax[1],
							ylabel='y',
							subtitle='fig2',
							xlabel='time (ms)',
							)
		finalizeFigure( fig,
						figSize=[6,4])
		
	"""

	# default input parameters
	params={	'xlabel':'',
				'ylabel':'',
				'title':'',
				'subtitle':'',
				'xlim':[],
				'ylim':[],
				'fontsize':8,
				'fontSizeTitle':None,
				'legendLoc':'best',
				'color':'grey',
				'linestyle':':',
				'alpha':1.0,
				'yticks':[],
				'ytickLabels':[],
				'xticks':[],
				'xtickLabels':[],
				'yAxisColor':'k',
				'legendOn':True,
				'ncol':1,
				'labelspacing':0.1,
				'numberLegendPoints':2,
				'handlelength':2,
				'tickDirection':'out',
				'originLines':True
				}
	
	# update default input parameters with kwargs
	params.update(kwargs)
	
	if params['fontSizeTitle']==None:
		params['fontSizeTitle']=params['fontsize']
	
	# check to see if ax is a list or array
	if type(ax)==list or type(ax)==_np.ndarray:
		pass
	else:
		ax=[ax]
		
	# allows it to handle a list of axes
	for i in range(0,len(ax)):
		
		# title and axis labels
		ax[i].set_ylabel(params['ylabel'],fontsize=params['fontsize'],color=params['yAxisColor'])
		if i==0:
			ax[i].set_title(params['title'],fontsize=params['fontSizeTitle'])
		if i==len(ax)-1:
			ax[i].set_xlabel(params['xlabel'],fontsize=params['fontsize'])
		
		# subtitle
		if params['subtitle']!='':
			subTitle(ax[i],params['subtitle'],fontSize=params['fontsize'])
		
		# legend
		if params['legendOn']==True:
			ax[i].legend(	fontsize=params['fontsize'],
						loc=params['legendLoc'],
						numpoints=params['numberLegendPoints'], # numpoints is the number of markers in the legend
						ncol=params['ncol'],
						labelspacing=params['labelspacing'],
						handlelength=params['handlelength']) 
				
		# set x and y axis tick label fontsize
		ax[i].tick_params(	axis='both',
						labelsize=params['fontsize'],
						direction=params['tickDirection']
						)
			
		# x and y limits
		if len(params['xlim'])>0:
			ax[i].set_xlim(params['xlim'])
		if len(params['ylim'])>0:
			ax[i].set_ylim(params['ylim'])
			
		# y ticks and y tick labels
		if params['yticks']!=[]:
			ax[i].set_yticks(params['yticks'])
		if params['ytickLabels']!=[]:
			ax[i].set_yticklabels(params['ytickLabels'])
			
		# x ticks and x tick labels
		if params['xticks']!=[]:
			ax[i].set_xticks(params['xticks'])
		if params['xtickLabels']!=[]:
			ax[i].set_xticklabels(params['xtickLabels'])
			
		ax[i].tick_params(axis='y', labelcolor=params['yAxisColor'])
			
		# add dotted lines along the x=0 and y=0 axes
		if params['originLines']==True:
			ax[i].axhline(y=0, color=params['color'],linestyle=params['linestyle'],alpha=params['alpha'])
			ax[i].axvline(x=0, color=params['color'],linestyle=params['linestyle'],alpha=params['alpha'])
		

		
	
		
		
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
	
	Examples
	--------
	Example1::
		
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
#%% shapes

def circle(ax,xy=(0,0),r=1,color='r',linestyle='-',alpha=1,fill=True,label=''):
	circle1 = _plt.Circle(xy, r, color=color,alpha=alpha,fill=fill,linestyle=linestyle)
	ax.add_artist(circle1)

def arrow(ax,xyTail=(0,0),xyHead=(1,1),color='r',width=3,headwidth=10,headlength=10,alpha=1.0,label=''):
	"""
	Draws an arrow
	
	References
	----------
	https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.annotate.html
	
	Examples
	--------
	Example1::
		
		t=_np.arange(0,10,2e-3)
		y1=_np.cos(_np.pi*2*t*2e-1)
		fig,ax=_plt.subplots(1,sharex=True)
		ax.plot(t,y1,label='y1')
		arrow(ax,xyTail=(0,0.0),xyHead=(0,1))
		arrow(ax,xyTail=(5,0.0),xyHead=(5,1))
		arrow(ax,xyTail=(10,0.0),xyHead=(10,1))
		arrow(ax,xyTail=(2.5,0.0),xyHead=(2.5,-1))
		arrow(ax,xyTail=(7.5,0.0),xyHead=(7.5,-1))
		finalizeSubplot( 	ax,
							ylabel='y',
							subtitle='fig1',
							title='title!',
							xlabel='t'
							)
		
	"""
	return ax.annotate("", xy=xyHead, xytext=xyTail, arrowprops=dict(	width=width,
														headwidth=headwidth,
														headlength=headlength,
														color=color,
														alpha=alpha,
														label=label),
				label=label)
	

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

def heatmap(dZ,vmin=None,vmax=None):
	"""
	I can't find a single good heatmap plotting library.
	matplotlib has pcolormesh and imshow, but both have issues.
	Seaborn has a heatmap() function (which is the best I've found), but the x
								   and y axes still have to be hacked to work
								   
	Example
	-------
	Example 1 ::
		
		x=_np.arange(10,200)
		y=_np.arange(-50,100)
		dZ=_pd.DataFrame(_np.random.rand(len(y),len(x)),index=y,columns=x)
		heatmap(dZ)
	"""
	# TODO work in in progress.  
	
# 	dZ=dZ.sort_index(ascending=False) # by default, the y axis is backwards
	import seaborn as sb
	fig,ax=_plt.subplots()
	
	sb.heatmap(dZ,vmin=vmin,vmax=vmax)
	
	ax.invert_yaxis()
	
	
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
		
		
### misc



import matplotlib as _mpl
def colorScalarMap(minValue=0,maxValue=10,colorMap=_mpl.cm.Reds):
	"""
	normalizes a matplotlib colormap to the desired range.  the returned colormap
	produces RGBA colors
	
	Parameters
	----------
	minValue : float
		lowest value for the colormap
	maxValue : float
		largest value for the colormap
	colorMap : matplotlib.colors.LinearSegmentedColormap
		matplotlib color map
		
	Returns
	-------
	s_m : matplotlib.cm.ScalarMappable
		normalized colormap. to convert a number to RGBA, call
		s_m.to_rgba(0.5) or with whatever number you wish
	
	Example
	-------
	:
		
		fig,ax=plt.subplots()
		x=np.arange(1,20,.1)
		y=np.cos(x)
		c=np.cos(x)
		cmap=colorScalarMap(-1,1)
		ax.scatter(x,y,color=cmap.to_rgba(c))
		plt.show()
		
	References
	----------
	https://stackoverflow.com/questions/48398726/flexible-way-to-add-subplots-to-figure-and-one-colorbar-to-figure
	"""
#	import matplotlib
	norm = _mpl.colors.Normalize(minValue,maxValue)
	s_m = _mpl.cm.ScalarMappable(cmap=colorMap, norm=norm)
	s_m.set_array([])
	return s_m


### custom colormaps


def RdOrWGnBu_colormap(plot=False):
	"""
	Red - Orange - White - Green - Blue colormap

	Parameters
	----------
	plot : TYPE, optional
		Optional plot to show an example of the new colormap.

	Returns
	-------
	newcmp : matplotlib.colors.ListedColormap
		new colormap
		
	Examples
	--------
	Example 1 ::
		
		_plt.close('all')
		cmap=RdOrWGnBu_colormap(plot=True)
		
	References
	----------
	::
		
		https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
		https://matplotlib.org/tutorials/colors/colormaps.html

	"""
	
	from matplotlib import cm
	from matplotlib.colors import ListedColormap, LinearSegmentedColormap
	import numpy as np

	top = cm.get_cmap('GnBu_r', 128)
	bottom = cm.get_cmap('OrRd', 128)
	
	newcolors = np.vstack((top(np.linspace(0, 1, 128)),
						   bottom(np.linspace(0, 1, 128))))
	newcmp = ListedColormap(newcolors, name='RdOrWGnBu')
		
	if plot==True:
		def plot_examples(cms):
			"""
			helper function to plot two colormaps
			"""
			x=np.linspace(0,2*np.pi,500)
			y=np.linspace(0,2*np.pi,500)
			X,Y=np.meshgrid(x,y)
			data=1*np.cos(2*X-5*Y)#+np.random.randn(X.shape[0],X.shape[1])*0.2
		
			fig, axs = _plt.subplots(1, len(cms), figsize=(2.5*len(cms), 3), constrained_layout=True)
			for [ax, cmap] in zip(axs, cms):
				psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=data.min(), vmax=data.max())
				ax.set_title(cmap.name)
				fig.colorbar(psm, ax=ax)
			_plt.show()
		
		viridis = cm.get_cmap('viridis', 256)
		Spectral = cm.get_cmap('Spectral_r', 256)
		seismic = cm.get_cmap('seismic', 256)

		plot_examples([viridis, Spectral, seismic, newcmp])
		
	return newcmp



#######################################################################################################
#%% data related


def plot_csv_file_that_is_actively_being_written_to(filename='out.csv', interval=5000, index='time'):
	""" 
	I don't get it.  This function works if I 'F9' the code, but not when I call it as a function.  No error provided so debugging is difficult. """
	import os
	
	if os.path.exists(filename):
		
		# init plot
		fig,ax=_plt.subplots()
		
		# update plot
		def update(i, a, ind, fn):
			print(i)
			a.clear()
			data=_pd.read_csv(fn).set_index(ind)
			data.plot(ax=a)
			_plt.show()
			
		# animations
		_animation.FuncAnimation(fig, update, interval=interval, fargs=(ax,index,filename,)) 
				
	else:
		print('File does not exist')