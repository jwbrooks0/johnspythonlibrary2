
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.signal import decimate as _decimate
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
import xarray as _xr
from deprecated import deprecated as _deprecated


# %% interpolating


def interpolate(da, x_new, plot=False, fill_value=_np.nan):
	""" wrapper for scipy's interp1d() function.  Works with xarray DataArrays and Datasets """
	
	from scipy.interpolate import interp1d
	
	# if DataArray
	if type(da) == _xr.core.dataarray.DataArray:
		
		dim = da.dims[0]
		x = da.coords[dim].values
		y = da.values
		interp_func = interp1d(x, y, fill_value=fill_value, bounds_error=False)
		da_new = _xr.DataArray(interp_func(x_new), dims=dim, coords=[x_new], attrs=da.attrs).dropna(list(da.dims)[-1])
		
		if plot is True:
			fig, ax = _plt.subplots()
			da.plot(ax=ax, label='original')
			da_new.plot(ax=ax, label='interpolated')
		
		return da_new
	
	# if Dataset, inerpolate each Dataset
	elif type(da) == _xr.core.dataset.Dataset:
		keys=list(da.keys())
		ds_new = _xr.Dataset()
		
		for key in keys:
			ds_new[key] = interpolate(da[key], x_new=x_new, plot=plot)

		return ds_new

# %% windowing

def hann_window_1D(array, hann_width, plot=False):
	"""
	
	Example
	-------
	Example 1 ::
		
		array = _np.random.rand(1000)
		hann_width = 500
		array_hann = hann_window_1D(array, hann_width, plot=True)

	Example 2 ::
		
		# xarray
		
		array = _xr.DataArray(_np.random.rand(1000), dims='t', coords=[np.arange(1000)*0.1])
		hann_width = 500
		array_hann = hann_window_1D(array, hann_width, plot=True)

	"""
	if type(array) == _xr.core.dataarray.DataArray:
		xr_array = array.copy()
		array = array.data
		data_xr = True
	else:
		data_xr = False
	
	from scipy.signal import hann
	if hann_width == array.shape[0]:
		y_hann = hann(hann_width)
	elif hann_width < array.shape[0]:
		y_hann = zero_pad(hann(hann_width), array.shape[0])
	else:
		raise Exception('Array should be longer than hann_width')
		
	y_out = y_hann * array
	
	if plot is True:
		fig, ax = _plt.subplots()
		ax.plot(array, label='Original data')
		ax.plot(y_hann, label='Hann window', linestyle='--', linewidth=2)
		ax.plot(y_out, label='Hann applied to data')
		ax.legend()
		
	if data_xr is True:
		xr_array.data = y_out
		y_out = xr_array
	
	return y_out


def hann_window_1D_to_2D_data(array, hann_width, axis=0, plot=False):
	"""
	Apply a 1D-hann window to a 2D dataarray along the specified axis
	
	Example
	-------
	Example 1 ::
		
		array = _np.random.rand(1000, 1200)
		hann_width = 500
		axis = 0
		array_hann = hann_window_1D_to_2D_data(array, hann_width, axis=axis, plot=True)

	Example 2 ::
		
		array = _np.random.rand(1000, 1200)
		hann_width = 500
		axis = 1
		array_hann = hann_window_1D_to_2D_data(array, hann_width, axis=axis, plot=True)

	Example 3 ::
		
		# xr example
		
		array = _xr.DataArray(_np.random.rand(1000, 1200), dims=['x', 'y'], coords=[_np.arange(1000), _np.arange(1200)])
		hann_width = 500
		axis = 1
		array_hann = hann_window_1D_to_2D_data(array, hann_width, axis=axis, plot=True)

	"""
	if type(array) == _xr.core.dataarray.DataArray:
		xr_array = array.copy()
		array = array.data
		data_xr = True
	else:
		data_xr = False
	
	from scipy.signal import hann
	if hann_width == array.shape[axis]:
		y_hann = hann(hann_width)
	elif hann_width < array.shape[axis]:
		y_hann = zero_pad(hann(hann_width), array.shape[axis])
	else:
		raise Exception('Array should be longer than hann_width')
		
	if axis == 0:
		y_hann = _np.repeat(y_hann[..., _np.newaxis], array.shape[1], axis=1)
	elif axis == 1:
		y_hann = _np.repeat(y_hann[_np.newaxis, ...], array.shape[0], axis=0)
		
	y_out = y_hann * array
	
	if plot is True:
		fig, ax = _plt.subplots(3, sharex=True)
		ax[0].contourf(array, label='Original data')
		ax[1].contourf(y_hann, label='Hann window', linestyle='--', linewidth=2)
		ax[2].contourf(y_out, label='Hann applied to data')

	if data_xr is True:
		xr_array.data = y_out
		y_out = xr_array
		
	return y_out


def zero_pad(array: _np.ndarray, target_length: int, axis: int = 0):
	"""
	Pads an array along a specified axis with zeros
	
	Example
	-------
	Example 1 ::
		
		# a 2D case where np.remainder(pad_size, 2) == 0
		
		array = _np.arange(5*7).reshape(5,7)
		axis = 1
		target_length = 11
		b = zero_pad(array, target_length=target_length, axis=axis)

	Example 2 ::
		
		# a 2D case where np.remainder(pad_size, 2) != 0
		
		array = _np.arange(5*7).reshape(5,7)
		axis = 1
		target_length = 12
		b = zero_pad(array, target_length=target_length, axis=axis)

	Example 3 ::
		
		# check the 1D case
		
		array = _np.arange(5) + 1
		axis = 0
		target_length = 10
		b = zero_pad(array, target_length=target_length, axis=axis)

	References
	----------
	 *  https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
	"""
	
	pad_size = (target_length - array.shape[axis]) / 2
	if pad_size <= 0:
		raise Exception('Target length must be larger than the array')

	if _np.remainder(pad_size, 1) == 0:
		pad_sizes = (int(pad_size), int(pad_size))
	else:
		pad_sizes = (int(pad_size), int(pad_size+1))
		
	npad = [(0, 0)] * array.ndim
	npad[axis] = pad_sizes

	return _np.pad(array, pad_width=npad, mode='constant', constant_values=0)


# %% downsampling

# @_deprecated(reason='Replaced by downsample().')
# def downsample_and_antialiasing(da, downsample_factor=10, plot=False):

# 	from scipy.signal import decimate #, resample
# 	
# 	t_orig = da.coords[da.dims[0]].data
# 	t_new = t_orig[::downsample_factor]
# 	da_new = _xr.DataArray(decimate(da.data, q=downsample_factor, ftype='fir'), dims=da.dims, coords=[t_new])
# 	
# 	if plot is True:
# 		fig, ax = _plt.subplots()
# 		da.plot(ax=ax, label='orig')
# 		da_new.plot(ax=ax, label='downsampled')
# 		ax.legend()
# 		
# 	return da_new


def downsample(da, downsample_factor=10, antialiasing=True, plot=False, axis=-1):
	"""
	

	Parameters
	----------
	da : xr.DataArray or xr.Dataset
		1D dataarray to be downsampled
	downsample_factor : int
		The factor to downsampled by.  
	antialiasing : bool
		Downsampling can be done with and without an antialiasing filter. The default is True (i.e. with antialiasing)
	plot : bool
		Optional plot of the results

	Returns
	-------
	da_new : xr.DataArray
		The downsampled dataarray

	Example
	-------
	Example 1 ::
		
		# single dataarray, process with and without antialiasing
		dt = 1e-6
		t = _np.arange(0, 1e4) * dt
		da = _xr.DataArray(_np.sin(2 * _np.pi * 1e3 * t) + _np.random.rand(len(t)))
		da_downsampled = downsample(da, downsample_factor=20, plot=True)
		da_downsampled = downsample(da, downsample_factor=20, plot=True, antialiasing=False)
		
		
	Example 2 ::
		
		# dataset containing two dataarrays
		dt = 1e-6
		t = _np.arange(0, 1e4) * dt
		da1 = _xr.DataArray(_np.sin(2 * _np.pi * 1e3 * t) + _np.random.rand(len(t)))
		da2 = _xr.DataArray(_np.sin(2 * _np.pi * 5e3 * t) + _np.random.rand(len(t)))
		ds = _xr.Dataset({'da1': da1, 'da2': da2})
		ds_downsampled = downsample(ds, downsample_factor=20, plot=True)
		
		
	
	"""
	if type(da) == _xr.core.dataset.Dataset:
		ds_new = _xr.Dataset()
		keys = list(da.keys())
		for key in keys:
			ds_new[key] = downsample(da[key], 
									downsample_factor=downsample_factor,
									antialiasing=antialiasing,
									plot=plot,
									axis=axis)
		return ds_new
	elif type(da) != _xr.core.dataarray.DataArray:
		raise Exception("Improper data type. ")
	
	# downsample time
	t_orig = da.coords[da.dims[axis]].data
	t_new = t_orig[::downsample_factor]
	
	# downsample data (with or without antialiasing)
	if antialiasing is True:
# 		coords = da.coords
		if axis == -1:
			axis = len(list(da.dims)) - 1
		coords = []
		for i, dim_key in enumerate(list(da.dims)):
			if axis == i:
				coords.append(t_new)
			else:
				coords.append(da.coords[dim_key])
		da_new = _xr.DataArray(_decimate(da.data, q=downsample_factor, ftype='fir', axis=axis), dims=da.dims, coords=coords)
	else:
		da_new = _xr.DataArray(da.data[::downsample_factor], dims=da.dims, coords=[t_new])
	
	# optional plot of results
	if plot is True:
		fig, ax = _plt.subplots()
		da.plot(ax=ax, label='orig')
		da_new.plot(ax=ax, label='downsampled')
		ax.legend()
		
	return da_new


# %% smoothing

def smooth_data(da, method='gaussian', method_parameters=dict(sigma=5), axis=-1, plot=False):
	"""
	Smooths data (low-pass filter).  Presently supports 'gaussian', 'boxcar'='uniform', and 'savgol'.
	This is a wrapper for several scipy-filters. 

	Parameters
	----------
	da : xarray DataArray
		Input data
	method : str, optional
		Smoothing method. The default is 'gaussian'.
	method_parameters : dict, optional
		Dictionary for parameters to pass to the scipy smoothing function. The default is dict(sigma=5).
	axis : int, optional
		Axis in da over which to apply the filter.  Meaning, this function can work on any axis. The default is -1.
	plot : bool
		Optional plot of results. The default is False.

	Returns
	-------
	da_smoothed : xarray DataArray
		Smoothed data

	"""
	
	# remove NaNs
	da.data[_np.isnan(da).data]=0
	
	# filter
	if method == 'gaussian': # method_parameters=dict(sigma=5)
		from scipy.ndimage import gaussian_filter1d
		da_smoothed = gaussian_filter1d(da, axis=axis, **method_parameters)
	elif method == 'boxcar':
		from scipy.ndimage import uniform_filter1d # method_parameters=dict(size=5)
		da_smoothed = uniform_filter1d(da, axis=axis, **method_parameters)
	elif method == 'savgol':
		from scipy.signal import savgol_filter # method_parameters=dict(window_length=10, polyorder=2)
		if _np.iscomplex(da[0]) is False:
			da_smoothed = savgol_filter(da, axis=axis, **method_parameters)
		else:
			da_smoothed = savgol_filter(da.real, axis=axis, **method_parameters) + 1j * savgol_filter(da.imag, axis=axis, **method_parameters)
			
	else:
		raise Exception('method not recognized')
	da_smoothed = _xr.DataArray(da_smoothed, dims=da.dims, coords=da.coords, attrs=da.attrs)
	
	if plot is True:
		fig, ax = _plt.subplots()
		da.plot(ax=ax, label='before')
		da_smoothed.plot(ax=ax, label='after')
	
	return da_smoothed
	

# %% low and highpass filtering

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
		da = da.rename({'time': 't'})
	if 't' not in da.dims:
		raise Exception('time not present')
		
	from scipy.signal import butter, filtfilt
	
	# construct butterworth filter
	samplingFreq = float(1.0 / (da.t[1] - da.t[0]))
	Wn = _np.array(cornerFreq).astype(float) / samplingFreq * 2  # I don't know why this factor of 2 needs to be here
	b, a = butter(	filterOrder, 
					Wn=Wn,
					btype=filterType,
					analog=False,
					)
	
	# perform forwards-backwards filter
	daOut = _xr.DataArray(	filtfilt(	b, a, da.values.reshape(-1),
										),
							   dims='t',
							   coords={'t': da.t})
	
	if plot is True:
		fig, ax = _plt.subplots()
		da.plot(ax=ax, label='original')
		daOut.plot(ax=ax, label='filtered')
		_finalizeSubplot(	ax,
							xlabel='Time',
							ylabel='Signal amplitude')
		
	return daOut


def boxcar_convolution_filter(	da, 
								width_in_time, 
								filter_type='low', 
								mode='reflect',
								plot=False):
	from scipy.ndimage import uniform_filter1d
		
	dt = _np.mean(da.t[1:].data - da.t[:-1].data)
	width = _np.round(width_in_time / dt).astype(int)
	da_lp = _xr.DataArray(uniform_filter1d(da.data, size=width, mode=mode), dims=da.dims, coords=da.coords)
	da_hp = da - da_lp.data
	
	if plot is True:
		fig, ax = _plt.subplots()
		da.plot(ax=ax, label='Original')
		da_lp.plot(ax=ax, label='Lowpass')
		da_hp.plot(ax=ax, label='Highpass')
		ax.legend()
		
	if filter_type == 'high':
		return da_hp
	else:
		return da_lp
	
		
def gaussianFilter(	da, 
				    timeFWHM, 
					filterType='high', 
					plot=False):
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
		da = da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('time not present')
	
	# import
	from scipy.ndimage import gaussian_filter1d
	
	# convert FWHM to standard deviation
	dt = float(da.t[1] - da.t[0])
	def fwhmToGaussFilterStd(fwhm, dt=dt):
		std = 1.0 / _np.sqrt(8 * _np.log(2)) * fwhm / dt
		return std
	std = fwhmToGaussFilterStd(timeFWHM, dt)
	
	# perform gaussian filter
	daFiltered=_xr.DataArray(	gaussian_filter1d(	da,
													std,
													axis=0,
													mode='nearest'),
								dims=['t'],
								coords={'t': da.t})

	# optional plot of results
	if plot is True:
		fig, ax = _plt.subplots()
		da.plot(ax=ax,label='Raw')
		daFiltered.plot(ax=ax, label='Low-pass')
		(da - daFiltered).plot(ax=ax, label='High-pass')
		_finalizeSubplot(	ax,
							xlabel='Time',
						    ylabel='Signal amplitude',)

	if filterType == 'low':
		return daFiltered
	elif filterType == 'high':
		return da-daFiltered
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
								coords={'t': time})
		else:
 			raise Exception('Invalid input type')
	if 'time' in da.dims:
		da = da.rename({'time': 't'})
	if 't' not in da.dims:
		raise Exception('time not present')
		
	from scipy.signal import butter, lfilter
 	
 	# construct butterworth filter
	samplingFreq = float(1.0 / (da.t[1] - da.t[0]))
	Wn = _np.array(cornerFreq).astype(float) / samplingFreq * 2  # I don't know why this factor of 2 needs to be here
	b, a = butter(	filterOrder, 
 					Wn=Wn,
 					btype=filterType,
 					analog=False,
 					)
	
	# perform forwards-backwards filter
	daOut = _xr.DataArray(	lfilter(b, a, da.values.reshape(-1)),
							dims='t',
							coords={'t': da.t})
	
	if plot is True:
		fig ,ax = _plt.subplots()
		da.plot(ax=ax, label='original')
		daOut.plot(ax=ax, label='filtered')
		_finalizeSubplot(	ax,
							xlabel='Time',
							ylabel='Signal amplitude',)
		
	return daOut
	