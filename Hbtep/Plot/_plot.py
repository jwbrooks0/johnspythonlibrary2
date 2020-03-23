
### Import

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import matplotlib as _mpl


import johnspythonlibrary2.Plot._plot as _plotMain
import johnspythonlibrary2.Hbtep.Get._get as _get

	
def stripeyPlot(	df,
					fig=None,
					ax=None,
					cax=None,
					**kwargs
					):
	"""
	Creates a standard stripey plot
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Dataframe containing the data.  The index is time (units in seconds and datatype as a float).  Columns are angles (units as degrees or radians and datatype as floats)
	fig : matplotlib.figure.Figure, optional
		optional figure on which to make the plot
	ax : matplotlib.axes._subplots.AxesSubplot, optional
		optional figure axis on which to make the plot
	cax : matplotlib.axes._axes.Axes, optional
		optional figire color axis on which to make the plot
		
	Returns
	-------
	fig : matplotlib.figure.Figure
		figure
	ax : matplotlib.axes._subplots.AxesSubplot
		axis
	cax : matplotlib.axes._axes.Axes
		color axis
	
	Example
	-------
	Example1::
		
		# generate fake data
		phi=_np.linspace(0,2*_np.pi,10)
		t=_np.arange(0,5e-3,2e-6)
		y=[]
		m=3
		for i,p in enumerate(phi):
			y.append(_np.cos(m*p+2*_np.pi*t*2e3+0.2*i))
		dfData=_pd.DataFrame(_np.array(y).transpose(),index=t,columns=phi)
			
		# functino call
		fig,ax,cax=stripeyPlot(dfData.copy()*1.1,title='asdf')

	Example2::
		
		shotno=98173
		_,df,dfMeta=_get.magneticSensorData(shotno,sensor='TA',tStart=1.5e-3,tStop=5e-3)
		angle=dfMeta.phi.values
		df.columns=angle
		df.index*=1e3
		fig,ax,cax=stripeyPlot(df*1e4,title='%d'%shotno,toroidal=True,subtitle='TA',xlabel='Time (ms)',ylabel=r'Toroidal angle, $\phi$')

	Example3::
		
		shotno=98173
		_,df,dfMeta=_get.magneticSensorData(shotno,sensor='PA1',tStart=1.5e-3,tStop=5e-3)
		angle=dfMeta.theta.values
		df.columns=angle
		df.index*=1e3
		fig,ax,cax=stripeyPlot(df*1e4,title='%d'%shotno,poloidal=True,fontsize=8,subtitle='PA1',xlabel='Time (ms)',ylabel=r'Poloidal angle, $\theta$')

		
	"""
	# default input parameters
	params={ 	'xlim':[],
				'ylim':[],
				'zlim':[],
				'yticks':[],
				'zticks':[],
				'ytickLabels':[],
				'levels':[],
				'colorMap':_mpl.cm.seismic,
				'xlabel':'Time',
				'ylabel':'Angle',
				'zlabel':'Amplitude',
				'title':'',
				'subtitle':'',
				'fontsize':8,
				'poloidal':False,
				'toroidal':False,
				}
	
	# update default input parameters with kwargs
	params.update(kwargs)
	
	# check inputs
	if 'float' not in df.columns.dtype.name or 'float' not in df.columns.dtype.name:
		raise Exception('Columns and indices should be floats.  ')
	if fig==None or ax==None or cax==None:
		fig,ax,cax=_plotMain.subplotsWithColormaps(1,True)
		
	# make a deep copy of the dataframe
	df=df.copy()
		
	# convert dependent axes to 2D
	angle=_np.copy(df.columns.values)	# make a deep copy of the columns
	if angle.max() < 10: 	# make sure angle is in units of degrees, not radians.
		angle*=180/_np.pi
	t=df.index.values
	ANGLE,T=_np.meshgrid(angle,t)
	
	# check parameters
	temp=df.abs().max().max()
	if params['yticks']==[]:
		if params['toroidal']==True:
			params['yticks']=[0,90,180,270,360]	
		elif params['poloidal']==True:
			params['yticks']=[-180,-90,0,90,180]
		else:
			pass
	if params['xlim']==[]:
		params['xlim']=[t[0],t[-1]]
	if params['ylim']==[]:
		params['ylim']=[angle.min(),angle.max()]
	if params['zlim']==[]:
		params['zlim']=[-temp,temp]
	if params['zticks']==[]:
		params['zticks']=_np.array([params['zlim'][0],params['zlim'][0]/2,0,params['zlim'][1]/2,params['zlim'][1]])
		if temp<1.0:
			params['ztickLabels']=['%.2e'%i for i in params['zticks']]
		else:
			params['ztickLabels']=['%.2f'%i for i in params['zticks']]
	if params['ytickLabels']==[]:
		if params['toroidal']==True:
			params['ytickLabels']=[r'$0^o$',r'$90^o$',r'$180^o$',r'$270^o$',r'$360^o$',]
		elif params['poloidal']==True:
			params['ytickLabels']=[r'$-180^o$',r'$-90^o$',r'$0^o$',r'$90^o$',r'$180^o$',]
		else:
			pass
	if params['levels']==[]:
		params['levels']=_np.linspace(-temp,temp,61)
	
	# create plot
	CS=ax.contourf(	T,
					ANGLE,
					df.values,
					levels=params['levels'],
					cmap=params['colorMap'],
					vmin=params['zlim'][0],
					vmax=params['zlim'][1],
					)
		

	# create colorbar
	cbar = _plt.colorbar(	CS,
						    ax=ax,
						    cax=cax,
							ticks=params['zticks'],
							pad=0.01,)
	
	# finalize plot
	cbar.ax.set_yticklabels(params['ztickLabels'],
							fontsize=params['fontsize'])
	cbar.ax.set_ylabel( '%s'%params['zlabel'],
						fontsize=params['fontsize'])
	_plotMain.finalizeSubplot(	ax,
								xlabel=params['xlabel'],
								subtitle=params['subtitle'],
								legendOn=False,
								yticks=params['yticks'],
								ytickLabels=params['ytickLabels'],
								ylabel=params['ylabel'],
								ylim=params['ylim'],
								xlim=params['xlim'],
								title=params['title'],
								fontsize=params['fontsize']
								)
	
	return fig,ax,cax