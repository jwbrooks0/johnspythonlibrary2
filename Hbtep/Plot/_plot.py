
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
# 	if angle.max() < 10: 	# make sure angle is in units of degrees, not radians.
# 		angle*=180/_np.pi
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
			params['ztickLabels']=['%.1f'%i for i in params['zticks']]
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




def modeContourPlot(	dfData,
						angles,
						fig=None,
						ax=None,
						cax=None,
						modeNumbers=_np.linspace(0.5,5.5,100),
						zlim=[0,8],
						title=''):
	"""
	calculates and then plots the amplitude of a range of non-integer poloidal mode
	numbers.  
	
	Parameters
	----------
	dfData : pandas.core.frame.DataFrame
		Dataframe containing the magnetic data.  E.g. PA1, TA.  The index is time (units in seconds and datatype as a float).  Columns are angles (units as degrees or radians and datatype as floats)
	angles : numpy array
		Angles (in radians) of associated with the sensor array
	fig : matplotlib.figure.Figure, optional
		optional figure on which to make the plot
	ax : matplotlib.axes._subplots.AxesSubplot, optional
		optional figure axis on which to make the plot
	cax : matplotlib.axes._axes.Axes, optional
		optional figire color axis on which to make the plot
	modeNumbers : numpy.ndarray
		Array of mode non-integer mode numbers to use in the analysis
	zlim : list of two floats
		This sets the colorbar scaling
	title : str
		(Optional) Title for the figure.
		
	Returns
	-------
	fig : matplotlib.figure.Figure
		figure
	ax : matplotlib.axes._subplots.AxesSubplot
		axis
	cax : matplotlib.axes._axes.Axes
		color axis
	
	
	Examples
	--------
	
		shotno=70463; zlim=[0,5.1]; tlim=[1.1e-3,5.6e-3]
		
		dfRaw,df,dfMeta=jpl2.Hbtep.Get.magneticSensorDataAll(	shotno,
																tStart=tlim[0],
																tStop=tlim[1],
																forceDownload=False)
		
		dfPA1=jpl2.Process.Pandas.filterDFByColOrIndex(df,'PA1')*1e4
		dfPA1Meta=jpl2.Process.Pandas.filterDFByColOrIndex(dfMeta,'PA1',False)
		
		mNumbers=np.linspace(0.5,5.5,100)

		fig,ax,cax=jpl2.Hbtep.Plot.modeContourPlot(dfPA1,dfPA1Meta.theta.values,modeNumbers=mNumbers,zlim=zlim,title='%d'%shotno)

	"""
	
	
	import matplotlib as _mpl
	
	def leastSquareModeAnalysis(	df,
									angles,
									modeNumbers=[0,-1,-2],
									timeFWHM_phaseFilter=0.1e-3,
									plot=False,
									title=''):
		"""
		Parameters
		----------
		df : pandas.core.frame.DataFrame
			Dataframe with multiple columns associated with different angles
			index = time
		angles : numpy.ndarray
			array of angles associated with the columns in df
		mode numbers : list of ints
			mode numbers to be analyzed
		timeFWHM_phaseFilter : float
			timewidth associated with pre-frequency calculating low-pass filter
			
		Returns
		-------
		dfResults : pandas.core.frame.DataFrame
			output
		"""
		
		# initialize
		n=len(angles)
		m=len(modeNumbers)*2
		if 0 in modeNumbers:
			m-=1
			
		# construct A matrix
		A=_np.zeros((n,m))
		i=0
		for mode in modeNumbers:
			if mode == 0:
				A[:,i]=1
				i+=1
			else:
				A[:,i]=_np.sin(mode*angles)
				A[:,i+1]=_np.cos(mode*angles)
				i+=2
		Ainv=_np.linalg.pinv(A)
		
		# perform least squares analysis	
		x=Ainv.dot(df.transpose().values)
		
		# calculate amplitudes, phases, frequencies, etc.
		dfResults=_pd.DataFrame(index=df.index)
		i=0
		for mode in modeNumbers:
				sin=x[i,:]
				cos=x[i+1,:]
				dfResults['%s'%mode]=_np.sqrt(sin**2+cos**2)

		return dfResults
	

	
	dfResults=_pd.DataFrame(index=dfData.index,columns=modeNumbers)
	for i,m in enumerate(modeNumbers):
#		print(i,m)
		dfResults[m]=leastSquareModeAnalysis(	dfData.copy(),
												angles,
												modeNumbers=[m])
	dfResults.index*=1e3
	
	fig,ax,cax=stripeyPlot(	dfResults,
								zlim=zlim,
								fig=fig,
								ax=ax,
								cax=cax,
#								colorMap=_mpl.cm.YlOrRd,
								colorMap=_mpl.cm.magma_r,
								ylabel='Poloidal mode \nnumber, m',
								zticks=_np.arange(zlim[0],zlim[-1]+0.5,2),
								ztickLabels=_np.arange(zlim[0],zlim[-1]+0.5,2),
								levels=_np.linspace(zlim[0],zlim[-1],41),
								xlabel='Time (ms)',
								zlabel=r'$ |\delta B_\theta |_m$ (G)',
#								zlabel='Mode\namplitude (G)',
								title=title,
# 								yticks=_np.arange(_np.floor(modeNumbers[0]),modeNumbers[-1],0.5)
								)
	t=dfResults.index.values
	for m in [1,2,3,4,5]:
		ax.plot([t[0],t[-1]],[m]*2,'k--',linewidth=0.5)
	
	return fig,ax,cax
