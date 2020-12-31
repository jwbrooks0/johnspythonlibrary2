
import numpy as _np
# import pandas as _pd
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
# from johnspythonlibrary2.Process.SigGen import chirp as _chirp
import xarray as _xr


def filtfilt(	da,
				cornerFreq,
				filterType='low',
				filterOrder=1,
				plot=False,
				axis='t',
				time=None):
	"""
	Forward-backwards filter using a butterworth filter
	
	Parameters
	----------
	da : xarray.core.dataarray.DataArray
		Signal.  Index is time with units in seconds.
	cornerFreq : float or numpy.array of floats
		Filter's corner frequency
		If bandpass or bandstop, cornerFreq is a numpy.array with two elements
	filterType : str
		* 'low' - low-pass filter 
		* 'high' - high-pass filter
		* 'bandpass' 
		* 'bandstop'
	filterOrder : int
		Order of the butterworth filter.  8 is default.
	plot : bool
		Optional plot
	time : numpy array, optional
		if da is a numpy array, then time needs to be provided
		
	Returns
	-------
	dfOut : xarray.core.dataarray.DataArray
		Filtered signal.  Index is time with units in seconds.
	
	Notes
	-----
	* Be careful with high filterOrders.  I've seen them produce nonsense results.  I recommend starting low and then turning up the order
		
	Examples
	--------
	Example 1::
		
		import numpy as np
		import xarray as xr
		t = np.linspace(0, 1.0, 2001)
		dt=t[1]-t[0]
		fs=1.0/dt
		xlow = np.sin(2 * np.pi * 5 * t)
		xmid = np.sin(2 * np.pi * 50 * t)
		xhigh = np.sin(2 * np.pi * 500 * t)
		x = xlow + xhigh + xmid
		da=xr.DataArray(x,dims=['t'],
				  coords={'t':t})
		dfOut=filtfilt( 	da,
							cornerFreq=30.0,
							filterType='low',
							filterOrder=8,
							plot=True)
		dfOut=filtfilt( 	da,
							cornerFreq=200.0,
							filterType='high',
							filterOrder=8,
							plot=True)
		dfOut=filtfilt( 	da,
							cornerFreq=np.array([25,150]),
							filterType='bandpass',
							filterOrder=4,
							plot=True)
		dfOut=filtfilt( 	x,
							cornerFreq=np.array([25,150]),
							filterType='bandpass',
							filterOrder=4,
							plot=True,
							time=t)

 	Example 2::
		
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		da=_xr.DataArray(y1,
 					  dims=['t'],
					   coords={'t':t})
		
		filterOrder=2
		cornerFreq=1000.0
		daHP=filtfilt(da,cornerFreq=cornerFreq,filterType='high',plot=False,filterOrder=filterOrder)
		daLP=filtfilt(da,cornerFreq=cornerFreq,filterType='low',plot=False,filterOrder=filterOrder)
		
		fig,ax=_plt.subplots()
		da.plot(ax=ax,label='Original')
		daHP.plot(ax=ax,label='Highpass')
		daLP.plot(ax=ax,label='Lowpass')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
 	Example 3::
		 
		import numpy as _np
		import xarray as _xr
		 
		t=_np.arange(0,100e-3-4e-6,2e-6)
		y=jpl2.Process.SigGen.gaussianNoise(t.shape,plot=False)
		da=_xr.DataArray(y,
 					  dims=['t'],
					   coords={'t':t})
		
		daOut=filtfilt( 	da,
										cornerFreq=np.array([1e5,1.5e5]),
										filterType='bandpass',
										filterOrder=4,
										plot=False)
		jpl2.Process.Spectral.fft(daOut,plot=True,
									trimNegFreqs=True)
		
		
	References
	----------
	* https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.filtfilt.html
	
	"""
	if type(da) not in [_xr.core.dataarray.DataArray]:
		if type(da) in [_np.ndarray]:
			da=_xr.DataArray(	da,
								dims=['t'],
								coords={'t':time})
		else:
			raise Exception('Invalid input type')
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('time not present')
		
	from scipy.signal import butter,filtfilt
	
	# construct butterworth filter
	samplingFreq=float(1.0/(da.t[1]-da.t[0]))
	Wn=_np.array(cornerFreq).astype(float)/samplingFreq*2  # I don't know why this factor of 2 needs to be here
	b, a = butter(	filterOrder, 
					Wn=Wn,
					btype=filterType,
					analog=False,
					)
	
	# perform forwards-backwards filter
	daOut = _xr.DataArray(	filtfilt(	b, a, da.values.reshape(-1),
										),
							   dims='t',
							   coords={'t':da.t})
	
	if plot==True:
		fig,ax=_plt.subplots()
		da.plot(ax=ax,label='original')
		daOut.plot(ax=ax,label='filtered')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
	return daOut
		
		
def gaussian(da,timeFWHM,filterType='high',plot=False):
	"""
	Low and highpass filters using scipy's gaussian convolution filter
	
	Parameters
	----------
	da  :  xarray.core.dataarray.DataArray
		unfiltered (raw) signal
	timeFWHM : float
		full width at half maximum of the gaussian with units in time.  this
		effectively sets the corner frequency of the filter
	filterType : str
		'high' - high-pass filter
		'low' - low-pass filter
	plot : bool
		plots the results
	plotGaussian : bool
		plots the gaussian distribution used for the filter
		
	Returns
	-------
	xarray.core.dataarray.DataArray
		filtered signal
		
	Examples
	--------
	Example 1::
		
		import xarray as xr
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		da=xr.DataArray(y1,
					  dims=['t'],
					  coords={'t':t}) 
		
		fwhm=0.5e-3
		daHP=gaussian(da,fwhm,'high',plot=False)
		daLP=gaussian(da,fwhm,'low',plot=False)
		
		fig,ax=_plt.subplots()
		ax.plot(da,label='Original')
		ax.plot(daHP,label='Highpass')
		ax.plot(daLP,label='Lowpass')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
		
	References
	----------
	* https://en.wikipedia.org/wiki/Full_width_at_half_maximum
	* https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.gaussian.html
	* https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
	
	"""
	
	# check input
	if type(da) not in [_xr.core.dataarray.DataArray]:
		raise Exception('Invalid input type')
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('time not present')
	
	# import
	from scipy.ndimage import gaussian_filter1d
	
	# convert FWHM to standard deviation
	dt=float(da.t[1]-da.t[0])
	def fwhmToGaussFilterStd(fwhm,dt=dt):
		std=1.0/_np.sqrt(8*_np.log(2))*fwhm/dt
		return std
	std=fwhmToGaussFilterStd(timeFWHM,dt)
	
	# perform gaussian filter
	daFiltered=_xr.DataArray(	gaussian_filter1d(	da,
													std,
													axis=0,
													mode='nearest'),
								dims=['t'],
								coords={'t':da.t})

	# optional plot of results
	if plot==True:
		fig,ax=_plt.subplots()
		da.plot(ax=ax,label='Raw')
		daFiltered.plot(ax=ax,label='Low-pass')
		(da-daFiltered).plot(ax=ax,label='High-pass')
		_finalizeSubplot(	ax,
							xlabel='Time',
						    ylabel='Signal amplitude',)

	if filterType=='low':
		return daFiltered
	elif filterType=='high':
		return da-daFiltered
	else:
		raise Exception('Bad filter type')


def gaussianFilter_df(df,timeFWHM,filterType='high',plot=False):
	"""
	Low and pass filters using scipy's gaussian convolution filter
	
	Parameters
	----------
	df  :  pandas.core.frame.DataFrame
		data = single or multiple array s
		index = time
	timeFWHM : float
		full width at half maximum of the gaussian with units in time.  this
		effectively sets the corner frequency of the filter
	filterType : str
		'high' - high-pass filter
		'low' - low-pass filter
	plot : bool
		plots the results
	plotGaussian : bool
		plots the gaussian distribution used for the filter
		
	Returns
	-------
	 : pandas.core.frame.DataFrame
		data = filtered verion of df
		same index and columns as df
		
	Examples
	--------
	Example1::
		
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		df=_pd.DataFrame(y1,
					  index=t,
					  columns=['orig']) 
		
		fwhm=0.5e-3
		dfHP=gaussianFilter_df(df,fwhm,'high',plot=False)
		dfHP.columns=['HP']
		dfLP=gaussianFilter_df(df,fwhm,'low',plot=False)
		dfLP.columns=['LP']
		
		fig,ax=_plt.subplots()
		ax.plot(df,label='Original')
		ax.plot(dfHP,label='Highpass')
		ax.plot(dfLP,label='Lowpass')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
		
	References
	----------
	https://en.wikipedia.org/wiki/Full_width_at_half_maximum
	https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.gaussian.html
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
	"""
	
	# import
	from scipy.ndimage import gaussian_filter1d
	import pandas as _pd
	
	# time step
	dt=df.index[1]-df.index[0]
	
	# convert FWHM to standard deviation
	def fwhmToGaussFilterStd(fwhm,dt=dt):
		std=1.0/_np.sqrt(8*_np.log(2))*fwhm/dt
		return std
	std=fwhmToGaussFilterStd(timeFWHM,dt)
	
	# perform gaussian filter
	dfFiltered=_pd.DataFrame(gaussian_filter1d(df,std,axis=0,mode='nearest'),index=df.index,columns=df.columns)

	# optional plot of results
	if plot==True:
		for i,(key,val) in enumerate(df.iteritems()):
			_plt.figure()
			_plt.plot(val.index,val,label='Raw')
			_plt.plot(dfFiltered[key].index,dfFiltered[key],label='Low-pass')
			_plt.plot(val.index,(df-dfFiltered)[key],label='High-pass')
			_plt.legend()
			_plt.title(key)

	if filterType=='low':
		return dfFiltered
	elif filterType=='high':
		return df-dfFiltered
	else:
		raise Exception('Bad filter type')

def butterworth(	da,
					cornerFreq,
					filterType='low',
					filterOrder=1,
					plot=False,
					time=None):
	"""
	Apply a butterworth filter to an input signal
	Introduces a frequency-dependent phase-shift on the output.  
	If no phase-shift is desired, use filtfilt() instead.  
	
	Parameters
	----------
	da : xarray.core.dataarray.DataArray
		Signal.  Index is time with units in seconds.
	cornerFreq : float or numpy.array of floats
		Filter's corner frequency
		If bandpass or bandstop, cornerFreq is a numpy.array with two elements
	filterType : str
		* 'low' - low-pass filter 
		* 'high' - high-pass filter
		* 'bandpass' 
		* 'bandstop'
	filterOrder : int
		Order of the butterworth filter.  8 is default.
	plot : bool
		Optional plot
	time : numpy array, optional
		if da is a numpy array, then time needs to be provided
		
	Returns
	-------
	daOut : xarray.core.dataarray.DataArray
		Filtered signal.  Index is time with units in seconds.
	
	Examples
	--------
 	Example 2::
		
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		from johnspythonlibrary2.Process.SigGen import chirp as _chirp
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		da=_xr.DataArray(y1,
 					  dims=['t'],
					   coords={'t':t})
		
		filterOrder=2
		cornerFreq=600.0
		daHP=butterworth(da,cornerFreq=cornerFreq,filterType='high',plot=False,filterOrder=filterOrder)
		daLP=butterworth(da,cornerFreq=cornerFreq,filterType='low',plot=False,filterOrder=filterOrder)
		
		fig,ax=_plt.subplots()
		da.plot(ax=ax,label='Original')
		daHP.plot(ax=ax,label='Highpass')
		daLP.plot(ax=ax,label='Lowpass')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
 	Example 3::
		 
		import numpy as _np
		import xarray as _xr
		 
		t=_np.arange(0,100e-3-4e-6,2e-6)
		from johnspythonlibrary2.Process.SigGen import gaussianNoise
		from johnspythonlibrary2.Process.Spectral import fft
		y=gaussianNoise(t.shape,plot=False)
		da=_xr.DataArray( 	y,
						    dims=['t'],
							coords={'t':t})
		
		daOut=butterworth( 	da,
							cornerFreq=_np.array([1e5,1.5e5]),
							filterType='bandpass',
							filterOrder=4,
							plot=False)
		fft(	daOut,
				plot=True,
				trimNegFreqs=True)
		
	"""
	
	# check input signal
	if type(da) not in [_xr.core.dataarray.DataArray]:
		if type(da) in [_np.ndarray]:
 			da=_xr.DataArray(	da,
								dims=['t'],
								coords={'t':time})
		else:
 			raise Exception('Invalid input type')
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('time not present')
		
	from scipy.signal import butter,lfilter
 	
 	# construct butterworth filter
	samplingFreq=float(1.0/(da.t[1]-da.t[0]))
	Wn=_np.array(cornerFreq).astype(float)/samplingFreq*2  # I don't know why this factor of 2 needs to be here
	b, a = butter(	filterOrder, 
 					Wn=Wn,
 					btype=filterType,
 					analog=False,
 					)
	
	# perform forwards-backwards filter
	daOut = _xr.DataArray(	lfilter(b, a, da.values.reshape(-1)),
							dims='t',
							coords={'t':da.t})
	
	if plot==True:
		fig,ax=_plt.subplots()
		da.plot(ax=ax,label='original')
		daOut.plot(ax=ax,label='filtered')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
	return daOut
	