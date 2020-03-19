
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt

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
