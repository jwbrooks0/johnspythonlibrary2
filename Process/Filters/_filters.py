
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.SigGen import chirp as _chirp


def filtfiltWithButterworth(	df,
								cornerFreq,
								filterType='low',
								filterOrder=1,
								plot=False):
	"""
	Forward-backwards filter using a butterworth filter
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Signal.  Index is time with units in seconds.
	cornerFreq : float
		Filter's corner frequency
	filterType : str
		'low' - low-pass filter 
		
		'high' - high-pass filter
		
	filterOrder : int
		Order of the butterworth filter.  8 is default.
	plot : bool
		Optional plot
		
	Returns
	-------
	dfOut : pandas.core.frame.DataFrame
		Filtered signal.  Index is time with units in seconds.
	
	Notes
	-----
	* Be careful with high filterOrders.  I've seen them produce nonsense results.  I recommend starting low and then turning up the order
		
	Examples
	--------
	Example1::
		
		import numpy as np
		t = np.linspace(0, 1.0, 2001)
		dt=t[1]-t[0]
		fs=1.0/dt
		xlow = np.sin(2 * np.pi * 5 * t)
		xhigh = np.sin(2 * np.pi * 250 * t)
		x = xlow + xhigh
		df=_pd.DataFrame(x,index=t,columns=['Original'])
		dfOut=filtfiltWithButterworth( 	df,
										cornerFreq=250.0,
										filterType='low',
										filterOrder=8,
										plot=True)
		dfOut=filtfiltWithButterworth( 	df,
										cornerFreq=250.0,
										filterType='high',
										filterOrder=8,
										plot=True)
		
	Example2::
		
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		df=_pd.DataFrame(y1,
					  index=t,
					  columns=['orig']) 
		
		filterOrder=1
		cornerFreq=900.0
		dfHP=filtfiltWithButterworth(df,cornerFreq=cornerFreq,filterType='high',plot=False,filterOrder=filterOrder)
		dfHP.columns=['HP']
		dfLP=filtfiltWithButterworth(df,cornerFreq=cornerFreq,filterType='low',plot=False,filterOrder=filterOrder)
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
	https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.filtfilt.html
	"""
	if filterType not in ['low','high']:
		raise Exception('Invalid filter type')
		
	from scipy.signal import butter,filtfilt
	
	samplingFreq=1.0/(df.index[1]-df.index[0])
	
	# construct butterworth filter
	Wn=float(cornerFreq)/samplingFreq
	b, a = butter(	filterOrder, 
					Wn=Wn,
					btype=filterType,
					analog=False,
					)
	
	# perform forwards-backwards filter
	dfOut = _pd.DataFrame(	filtfilt(	b, a, df.values.reshape(-1),
										),
							df.index,
							columns=['%spassFiltered'%filterType])
	
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(df,label='original')
		ax.plot(dfOut,label='filtered')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
	return dfOut
	
	



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


def butterworthFilter(	df,
						cornerFreq,
						filterType='low',
						filterOrder=1,
						plot=False):
	"""
	Butterworth filter, low or high-pass.  This imposes a phase shift on the signal.
	Use filtfilt or gaussian to avoid this.
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Signal.  Index is time with units in seconds.
	cornerFreq : float
		Filter's corner frequency
	filterType : str
		'low' - low-pass filter 
		
		'high' - high-pass filter
		
	filterOrder : int
		Order of the butterworth filter.  8 is default.
	plot : bool
		Optional plot
	
	Returns
	-------
	dfOut : pandas.core.frame.DataFrame
		Filtered signal.  Index is time with units in seconds.
	
	Notes
	-----
	* Be careful with high filterOrders.  I've seen them produce nonsense results.  I recommend starting low and then turning up the order
	* This imposes a phase shift on the original signal.  Use a filtfilt filter to get no phase shift.
		
	Examples
	--------
	Example1::
		
		import numpy as np
		t = np.linspace(0, 1.0, 2001)
		dt=t[1]-t[0]
		fs=1.0/dt
		xlow = np.sin(2 * np.pi * 5 * t)
		xhigh = np.sin(2 * np.pi * 250 * t)
		x = xlow + xhigh
		df=_pd.DataFrame(x,index=t,columns=['Original'])
		dfOut=butterworthFilter( 	df,
										cornerFreq=250.0,
										filterType='low',
										filterOrder=8,
										plot=True)
		dfOut=butterworthFilter( 	df,
										cornerFreq=250.0,
										filterType='high',
										filterOrder=8,
										plot=True)
		
	Example2::
		
		t=_np.arange(0,100e-3-4e-6,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[10e-3,90e-3],[fStart,fStop])#[1e-3,19.46e-3]
		df=_pd.DataFrame(y1,
					  index=t,
					  columns=['orig']) 
		
		filterOrder=1
		cornerFreq=900.0
		dfHP=butterworthFilter(df,cornerFreq=cornerFreq,filterType='high',plot=False,filterOrder=filterOrder)
		dfHP.columns=['HP']
		dfLP=butterworthFilter(df,cornerFreq=cornerFreq,filterType='low',plot=False,filterOrder=filterOrder)
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
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
	https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
	"""	
	
	from scipy.signal import butter, lfilter
	
	if filterType not in ['low','high']:
		raise Exception('Invalid filter type')
	
	samplingFreq=1.0/(df.index[1]-df.index[0])
	
	# construct butterworth filter
	Wn=float(cornerFreq)/samplingFreq*2  #TODO why is there a 2 here?  double check this.
	b, a = butter(	filterOrder, 
					Wn=Wn,
					btype=filterType,
					analog=False,
					)
	dfOut = _pd.DataFrame(	lfilter(b, a, df.values.reshape(-1)),
							index=df.index,
							columns=['%spassFiltered'%filterType])
	
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(df,label='original')
		ax.plot(dfOut,label='filtered')
		_finalizeSubplot(ax,
					   xlabel='Time',
					   ylabel='Signal amplitude',)
		
		
		
	return dfOut