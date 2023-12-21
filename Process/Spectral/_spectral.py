
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from scipy import fftpack as _fftpack
from scipy.signal import welch as _welch
try: # note that scipy changed the location of their _spectral_helper function
	from scipy.signal.spectral import _spectral_helper
except ImportError as e:
	from scipy.signal._spectral_py import _spectral_helper
from johnspythonlibrary2 import Plot as _plot
from johnspythonlibrary2.Plot import subTitle as _subTitle, finalizeFigure as _finalizeFigure, finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Misc import check_dims as _check_dims
# from johnspythonlibrary2.Process.Spectral import fft as _fft
import xarray as _xr
from deprecated import deprecated



###############################################################################
#%% Fourier methods

def signal_spectral_properties(da,nperseg=None,verbose=True):
# 	print('work in progress')
	
	# check input
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'%(str(da.dims)))
	
	# preliminary steps
	params={}
	params['dt']=float(da.t[1]-da.t[0])
	params['f_s']=1.0/params['dt']
	if type(nperseg) != int:
		nperseg=da.t.shape[0]
	if verbose: print('Window size: %d'%nperseg)
	
	# Nyquist frequency (highest frequency)
	f_nyquist=params['f_s']/2.
	params['f_nyquist']=f_nyquist
	if verbose: print("Nyquist freq., %.2f" % f_nyquist)
	
	# time window
	time_window=params['dt']*nperseg
	params['time_window']=time_window
	if verbose: print("Time window: %.3e s"%time_window)
	
# 	# lowest frequency to get a full wavelength
# 	params['f_min']=1./(nperseg*params['dt'])
# 	if verbose: print("Lowest freq. to get 1 wavelength, %.2f" % params['f_min'] )

	# frequency resolution, also the lowest frequency to get a full wavelength
	params['f_res']=params['f_s']/nperseg
	if verbose: print("Frequency resolution, %.2f" % params['f_res'] )

	return params #dt, f_s, f_nyquist, time_window, f_res


# def fft_max_freq(df_fft,positiveOnly=True):
#  	"""
#  	Calculates the maximum frequency associted with the fft results
 	
#  	Example
#  	-------
#  	::
# 		
# 		import numpy as np
# 		
# 		dt=2e-6
# 		f=1e3
# 		t=np.arange(0,10e-3,dt)
# 		y=np.sin(2*np.pi*t*f)
# 		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
#  							index=t,
#  							columns=['a','b','c'])
# 		df_fft=fft_df(df,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
# 		max_freq=fft_max_freq(df_fft)
# 		
#  	"""
#  	df_fft=_np.abs(df_fft.copy())
 	
#  	if positiveOnly==True:
# 		df_fft=df_fft[df_fft.index>0]
 	
#  	return df_fft.idxmax(axis=0)


def fft_max_freq(da_fft,positiveOnly=True):
	"""
	Calculates the maximum frequency associted with the fft results
	
	Parameters
	----------
	da_fft : pandas dataframe or xarray dataarray
	
	Example
	-------
	Example - pandas ::
		
		import numpy as np
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
							index=t,
							columns=['a','b','c'])
		df_fft=fft_df(df,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
		# DataFrame
		max_freq=fft_max_freq(df_fft,positiveOnly=True)
		print(max_freq)
		# Series
		max_freq=fft_max_freq(df_fft['a'],positiveOnly=True)
		print('max frequency = %.3f'%max_freq)
		
		
	Example - xarray ::
		
		dt=2e-6
		f=1e3
		t=_np.arange(0,10e-3,dt)
		y=_np.sin(2*_np.pi*t*f)
		da=_xr.DataArray( 	y,
							dims=['t'],
							coords={'t':t})
		da_fft=fft(da,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
		max_freq=fft_max_freq(da_fft,positiveOnly=True)
		print('max frequency = %.3f'%max_freq)
		
	"""
	da_fft=_np.abs(da_fft.copy())
	
	if type(da_fft)==_xr.core.dataarray.DataArray:
		
		if positiveOnly==True:
			da_fft=da_fft[da_fft.f>=0]
		
		return float(da_fft.idxmax().data)
	
	elif type(da_fft)==_pd.core.frame.DataFrame or type(da_fft)==_pd.core.frame.Series:
		
		if positiveOnly==True:
			da_fft=da_fft[da_fft.index>0]
			
		return da_fft.idxmax(axis=0)
	
	else:
		raise Exception('invalid input type')
	
	
# # depricated
# def fft_df(df,plot=False,trimNegFreqs=False,normalizeAmplitude=False):
# 	"""
# 	Simple wrapper for fft from scipy
# 	
# 	Parameters
# 	----------
# 	df : pandas.core.frame.DataFrame
# 		dataframe of time dependent data
# 		index = time
# 	plot : bool
# 		(optional) Plot results
# 	trimNegFreqs : bool
# 		(optional) True - only returns positive frequencies
# 	normalizeAmplitude : bool
# 		(optional) True - normalizes the fft (output) amplitudes to match the time series (input) amplitudes
# 		
# 	Returns
# 	-------
# 	dfFFT : pandas.core.frame.DataFrame
# 		complex FFT of df
# 	
# 	References
# 	-----------
# 	https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
# 	
# 	Example
# 	-------
# 	::
# 		
# 		import numpy as np
# 		
# 		dt=2e-6
# 		f=1e3
# 		t=np.arange(0,10e-3,dt)
# 		y=np.sin(2*np.pi*t*f)
# 		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
# 							index=t,
# 							columns=['a','b','c'])
# 		df_fft=fft_df(df,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
# 		
# 	"""
# 	
# 	if type(df)!=_pd.core.frame.DataFrame:
# 		if type(df)==_pd.core.series.Series:
# 			df=_pd.DataFrame(df)
# 		else:
# 			raise Exception('Input data not formatted correctly')
# 	
# 	# initialize
# 	
# 	# fft
# 	from numpy.fft import fft
# 	dt=df.index[1]-df.index[0]
# 	freq = _fftpack.fftfreq(df.shape[0],d=dt)
# 	dfFFT=df.apply(fft,axis=0).set_index(freq)
# # 	dfFFT=df.apply(_fftpack.fft,axis=0).set_index(freq)
# 	
# 	# options
# 	if normalizeAmplitude==True:
# 		N=df.shape[0]
# 		dfFFT*=1.0/N # 2/N if negative freqs have already been trimmed
# 	if trimNegFreqs==True:
# 		dfFFT=dfFFT[dfFFT.index>=0]
# 		
# 	# optional plot of results
# 	if plot==True:
# 		
# 		dfAmp=dfFFT.abs()
# 		dfPhase=phase_df(dfFFT)
# 		for i,(key,val) in enumerate(df.iteritems()):
# # 			f,(ax1,ax2,ax3)=_plt.subplots(nrows=32)
# 			f,(ax1,ax2)=_plt.subplots(nrows=2)
# 			
# 			ax1.plot(val)
# 			ax1.set_ylabel('Orig. signal')
# 			ax1.set_xlabel('Time')
# 			ax1.set_title(key)
# 			
# 			ax2.loglog(dfPhase.index,dfAmp[key],marker='.')
# 			ax2.set_ylabel('Amplitude')
# 			
# # 			ax3.plot(dfPhase.index,dfPhase[key],marker='.',linestyle='')
# # 			ax3.set_ylabel('Phase')
# 			ax2.set_xlabel('Frequency')
# # 			ax3.set_ylim([-_np.pi,_np.pi])
# 		
# 	# return results
# 	return dfFFT


	
# def fft(	da,
# 			plot=False,
# 			trimNegFreqs=False,
# 			normalizeAmplitude=False,
# 			sortFreqIndex=False,
# 			returnAbs=False,
# 			zeroTheZeroFrequency=False,
# 			realAmplitudeUnits=False,
# 			fft_scale='log'):
# 	#TODO update this to allow for theta units (not just time)
# 	"""
# 	Simple wrapper for fft from scipy
# 	
# 	Parameters
# 	----------
# 	da : xarray.DataArray or xarray.Dataset
# 		dataarray of time dependent data
# 		coord1 = time or t (units in seconds)
# 	plot : bool
# 		(optional) Plot results
# 	trimNegFreqs : bool
# 		(optional) True - only returns positive frequencies
# 	normalizeAmplitude : bool
# 		(optional) True - normalizes the fft (output) amplitudes to match the time series (input) amplitudes
# 		
# 	Returns
# 	-------
# 	da_fft : xarray.DataArray
# 		complex FFT of da
# 	
# 	References
# 	-----------
# 	https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
# 	
# 	Examples
# 	--------
# 	Example 1::
# 		
# 		import numpy as np
# 		import xarray as xr
# 		
# 		dt=2e-6
# 		f=1e3
# 		t=np.arange(0,10e-3,dt)
# 		y=np.sin(2*np.pi*t*f)+np.random.normal(0,1,t.shape)
# 		da=xr.DataArray( 	y,
# 							dims=['t'],
# 							coords={'t':t})
# 		fft_result=fft(da,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
# 		
# 		
# 	Example 2::
# 		
# 		import numpy as np
# 		import xarray as xr
# 		
# 		dt=2e-6
# 		f1=1e3
# 		f2=1.3e4
# 		f3=3.3e4
# 		t=np.arange(0,10e-3,dt)
# 		y1=np.sin(2*np.pi*t*f1)+np.random.normal(0,1,t.shape)
# 		y2=np.sin(2*np.pi*t*f2+np.pi/2.0)+np.random.normal(0,1,t.shape)
# 		y3=np.sin(2*np.pi*t*f3+np.pi/2.0)+np.random.normal(0,1,t.shape)
# 		da1=xr.DataArray( 	y1,
# 							dims=['t'],
# 							coords={'t':t},
# 							name='y1')
# 		da2=xr.DataArray( 	y2,
# 							dims=['t'],
# 							coords={'t':t},
# 							name='y2')
# 		da3=xr.DataArray( 	y3,
# 							dims=['t'],
# 							coords={'t':t},
# 							name='y3')
# 		ds=	xr.Dataset({'da1':da1,
# 			  'da2':da2,
# 			  'da3':da3})
# 		fft_result=fft(	ds,
# 						plot=True,
# 						trimNegFreqs=True,
# 						normalizeAmplitude=False)
# 				
# 	"""
# 	import xarray as xr
# 	from numpy.fft import fft as fft_np
# # 	from scipy.fft import fft as fft_np
# 	
# 	# check input
# 	if type(da) not in [xr.core.dataarray.DataArray,xr.core.dataset.Dataset]:
# 		raise Exception('Input data not formatted correctly')
# 	if type(da) in [xr.core.dataarray.Dataset]:
# 		return da.apply(fft,
# 						plot=plot,
# 						trimNegFreqs=trimNegFreqs,
# 						normalizeAmplitude=normalizeAmplitude,
# 						sortFreqIndex=sortFreqIndex,
# 						realAmplitudeUnits=realAmplitudeUnits,
# 						zeroTheZeroFrequency=zeroTheZeroFrequency)

# 	try: 
# 		time=_np.array(da.t)
# 	except:
# 		raise Exception('Time dimension needs to be labeled t')
# 	
# 	# do fft
# 	dt=time[1]-time[0]
# 	freq = _fftpack.fftfreq(da.t.shape[0],d=dt)
# 	fft_results=xr.DataArray(	fft_np(da.data),
# 								dims=['f'],
# 								coords={'f':freq})
# 	fft_results.attrs["units"] = "au"
# 	fft_results.f.attrs["units"] = "Hz"
# 	fft_results.f.attrs["long_name"] = 'Frequency'
# 	fft_results.attrs["long_name"] = 'FFT amplitude'
# 	
# 	# options
# 	if realAmplitudeUnits==True:
# 		N=da.t.shape[0]
# 		fft_results*=2.0/N 
# 	if trimNegFreqs==True:
# 		fft_results=fft_results.where(fft_results.f>=0).dropna(dim='f')
# 	if sortFreqIndex == True:
# 		fft_results=fft_results.sortby('f')
# 	if returnAbs == True:
# 		fft_results=_np.abs(fft_results)
# 	if zeroTheZeroFrequency == True:
# 		fft_results.loc[0] = 0	
# 	elif normalizeAmplitude==True:
# 		fft_results/=fft_results.sum()
# 		
# 	# optional plot of results
# 	if plot==True:
# 		_fftPlot(fft_results,kwargs={'s':da},fft_scale=fft_scale)
# 		
# 	return fft_results


def fft(	da,
			plot=False,
			trimNegFreqs=False,
			normalizeAmplitude=False,
			sortFreqIndex=False,
			returnAbs=False,
			zeroTheZeroFrequency=False,
			realAmplitudeUnits=False,
			fft_scale='log',
			fft_units='Hz',
			fft_dim_name='f',
			verbose=False):
	"""
	Simple wrapper for fft from scipy
	
	Parameters
	----------
	da : xarray.DataArray or xarray.Dataset
		dataarray of time dependent data
		if dataset, the function is applied to each dataarray
	plot : bool
		(optional) Plot results
	trimNegFreqs : bool
		(optional) True - only returns positive frequencies
	normalizeAmplitude : bool
		(optional) True - normalizes the fft (output) amplitudes to match the time series (input) amplitudes
	sortFreqIndex : bool
		Sort the frequency coordinate in the dataarray in ascending order
	returnAbs : bool
		Returns the absolute value of the FFT results
	zeroTheZeroFrequency : bool
		Zeros the value at f=0
	realAmplitudeUnits : bool
		Attempts to adjust the amplitude so that it returns physical units
	fft_scale : str
		The y-scale for the fft plot
	verbose : str
		Prints misc. details of the function to screen
		
	Returns
	-------
	da_fft : xarray.DataArray
		complex FFT of da
	
	References
	-----------
	https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
	
	Examples
	--------
	Example 1::
		
		import numpy as np
		import xarray as xr
		
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)+np.random.normal(0,1,t.shape)
		da=xr.DataArray( 	y,
							dims=['t'],
							coords={'t':t})
		fft_result=fft(da,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
		
		
	Example 2::
		
		import numpy as np
		import xarray as xr
		
		dt=2e-6
		f1=1e3
		f2=1.3e4
		f3=3.3e4
		t=np.arange(0,10e-3,dt)
		y1=np.sin(2*np.pi*t*f1)+np.random.normal(0,1,t.shape)
		y2=np.sin(2*np.pi*t*f2+np.pi/2.0)+np.random.normal(0,1,t.shape)
		y3=np.sin(2*np.pi*t*f3+np.pi/2.0)+np.random.normal(0,1,t.shape)
		da1=xr.DataArray( 	y1,
							dims=['t'],
							coords={'t':t},
							name='y1')
		da2=xr.DataArray( 	y2,
							dims=['t'],
							coords={'t':t},
							name='y2')
		da3=xr.DataArray( 	y3,
							dims=['t'],
							coords={'t':t},
							name='y3')
		ds=	xr.Dataset({'da1':da1,
			  'da2':da2,
			  'da3':da3})
		fft_result=fft(	ds,
						plot=True,
						trimNegFreqs=True,
						normalizeAmplitude=False)
				
	"""
	import xarray as xr
	from numpy.fft import fft as fft_np
 	# from scipy.fft import fft as fft_np
	
	# check input
	if type(da) not in [xr.core.dataarray.DataArray,xr.core.dataset.Dataset]:
		raise Exception('Input data not formatted correctly')
	if type(da) in [xr.core.dataarray.Dataset]:
		return da.apply(fft,
						plot=plot,
						trimNegFreqs=trimNegFreqs,
						normalizeAmplitude=normalizeAmplitude,
						sortFreqIndex=sortFreqIndex,
						realAmplitudeUnits=realAmplitudeUnits,
						zeroTheZeroFrequency=zeroTheZeroFrequency)

	x = da.coords[da.dims[0]]
		
	if verbose==True:
		signal_spectral_properties(da, verbose=True)
	
	# do fft
	dx = float(x[1] - x[0])
	freq = _fftpack.fftfreq(len(x), d=dx)
	fft_results=xr.DataArray(	fft_np(da.data),
								dims=[fft_dim_name],
								coords=[freq])
	fft_results.attrs["units"] = "au"
	fft_results.f.attrs["units"] = fft_units
	fft_results.f.attrs["long_name"] = 'Frequency'
	fft_results.attrs["long_name"] = 'FFT amplitude'
	
	# options
	if realAmplitudeUnits is True:
		N=da.t.shape[0]
		fft_results*=2.0/N 
	if trimNegFreqs is True:
		fft_results=fft_results.where(fft_results.f>=0).dropna(dim='f')
	if sortFreqIndex is True:
		fft_results=fft_results.sortby(fft_dim_name)
	if returnAbs is True:
		fft_results=_np.abs(fft_results)
	if zeroTheZeroFrequency is True:
		fft_results.loc[0] = _np.nan	
		# fft_results.loc[0] = 0	
	elif normalizeAmplitude is True:
		fft_results/=fft_results.sum()
		
	# optional plot of results
	if plot==True:
		_fftPlot(da, fft_results,fft_scale=fft_scale)
		
	return fft_results



def fft_average(	da,
					nperseg=None,
					noverlap=None,
					plot=False,
					verbose=True,
					trimNegFreqs=False,
					normalizeAmplitude=False,
					sortFreqIndex=False,
					returnAbs=False,
					zeroTheZeroFrequency=False,
					realAmplitudeUnits=False,
					f_units='Hz',
					fft_scale='log',
					fig=None):
	"""
	Computes an averaged abs(fft) using Welch's method.  This is mostly a wrapper for scipy.signal.welch
	
	Parameters
	----------
	da : xarray.DataArray or xarray.Dataset
		dataarray of time dependent data
		coord1 = time or t (units in seconds)
	nperseg : int, optional
		Length of each segment. Defaults to None, but if window is str or tuple, is set to 256, and if window is array_like, is set to the length of the window.
	noverlap : int, optional
		Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
	plot : bool
		(optional) Plot results
	trimNegFreqs : bool
		(optional) True - only returns positive frequencies

	Returns
	-------
	da_fft : xarray.DataArray
		averaged abs(FFT) of da
	
	References
	-----------
	* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
	
	Examples
	--------
	Example 1::
		
		import numpy as np
		import xarray as xr
		
		dt=2e-6
		f=2e4
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)+np.random.normal(0,1,t.shape)
		da=xr.DataArray( 	y,
							dims=['t'],
							coords={'t':t})
		fft_result=fft_average(	da,
								plot=True,
								trimNegFreqs=False,
								nperseg=1000,
								noverlap=0)
		
		
	Example 2::
		
		import numpy as np
		import xarray as xr
		
		dt=2e-6
		f1=1e3
		f2=1.3e4
		f3=3.3e4
		t=np.arange(0,10e-3,dt)
		y1=np.sin(2*np.pi*t*f1)+np.random.normal(0,1,t.shape)
		y2=np.sin(2*np.pi*t*f2+np.pi/2.0)+np.random.normal(0,1,t.shape)
		y3=np.sin(2*np.pi*t*f3+np.pi/2.0)+np.random.normal(0,1,t.shape)
		da1=xr.DataArray( 	y1,
							dims=['t'],
							coords={'t':t},
							name='y1')
		da2=xr.DataArray( 	y2,
							dims=['t'],
							coords={'t':t},
							name='y2')
		da3=xr.DataArray( 	y3,
							dims=['t'],
							coords={'t':t},
							name='y3')
		ds=	xr.Dataset({'da1':da1,
			  'da2':da2,
			  'da3':da3})
		fft_result=fft_average(	ds,
								plot=True,
								trimNegFreqs=False,
								nperseg=500,
								noverlap=0)
		
	Example 3 ::
		
		# quick test of the "Real amplitude" feature 
		
		t=np.arange(0,1e0,1e-6)
		f1=1000
		y1=np.sin(2*np.pi*f1*t)
		f2=10000
		y2=np.sin(2*np.pi*f2*t)
		da=xr.DataArray( 	y1+y2,
							dims='t',
							coords=[t])
		
		fft(da,plot=True,fft_scale='linear',realAmplitudeUnits=True)
		fft_average(da,plot=True,fft_scale='linear',realAmplitudeUnits=True,nperseg=t.shape[0]//10)

				
	"""
	import xarray as xr
	
	# check input
	if type(da) not in [xr.core.dataarray.DataArray,xr.core.dataset.Dataset]:
		raise Exception('Input data not formatted correctly')
	if type(da) in [xr.core.dataarray.Dataset]:
		return da.apply(fft_average,
						nperseg=nperseg,
						noverlap=noverlap,
						plot=plot,
						trimNegFreqs=trimNegFreqs,
						normalizeAmplitude=normalizeAmplitude,
						sortFreqIndex=sortFreqIndex,
						realAmplitudeUnits=realAmplitudeUnits,
						zeroTheZeroFrequency=zeroTheZeroFrequency,
						verbose=verbose,
						fft_scale=fft_scale)

	try: 
		time=_np.array(da.t)
	except:
		raise Exception('Time dimension needs to be labeled t')
	
	signal_spectral_properties(da, nperseg=nperseg, verbose=verbose)
	
	# do fft
	from scipy.signal import welch
	dt=time[1]-time[0]
	freq, fft_abs=	welch(	da.data,
							fs=1.0/dt,
							nperseg=nperseg,
							noverlap=noverlap,
							return_onesided=trimNegFreqs,
 							scaling='spectrum',
							)
	fft_results=xr.DataArray(	fft_abs,
								dims=['f'],
								coords={'f':freq})
	fft_results.attrs["units"] = "au"
	fft_results.f.attrs["units"] = f_units
	fft_results.f.attrs["long_name"] = 'Frequency'
	fft_results.attrs["long_name"] = 'FFT amplitude'
	
	# options
	if realAmplitudeUnits==True:
		N=da.t.shape[0]
		fft_results*=4  # why is there a 4 here???  I was expecting a 2/N or 2.  
	if trimNegFreqs==True:
		fft_results=fft_results.where(fft_results.f>=0).dropna(dim='f')
	if sortFreqIndex == True:
		fft_results=fft_results.sortby('f')
	if returnAbs == True:
		fft_results=_np.abs(fft_results)
	if zeroTheZeroFrequency == True:
		fft_results.loc[0] = 0	
	elif normalizeAmplitude==True:
		fft_results/=fft_results.sum()
		
	# optional plot of results
	if plot==True:
		_fftPlot(da, fft_results,fft_scale=fft_scale, fig=fig)
		
	return fft_results
	
	
	
def _fftPlot(da_orig, da_fft, fft_scale='log', fig=None):

	if 'f' in da_fft.dims:
		da_temp=_np.abs(da_fft.copy()).sortby('f')
	else:
		da_temp=_np.abs(da_fft.copy()).sortby('m')
		
	if type(fig)==type(None):
		fig,(ax1,ax2)=_plt.subplots(nrows=2)
	else:
		ax1,ax2=fig.get_axes()
	da_orig.plot(ax=ax1)
	
	da_temp.plot(ax=ax2)
	ax1.set_ylabel('Orig. signal')
# 	ax1.set_xlabel('Time')
	ax2.set_yscale(fft_scale)
	ax1.set_title(da_temp.name)
 		# ax2.set_xscale('log')
	ax2.set_ylabel('FFT Amplitude')
# 	ax2.set_xlabel('Frequency')
	
	return fig,(ax1,ax2)


def fftSingleFreq(da,f,plot=False):
	"""
	Performs a Fourier transform of a signal at a single frequency
	
	Parameters
	----------
	da : xarray.DataArray
		the signal being analyzed
	f : float
		The frequency (in Hz) that is being investigated
		
	Return
	------
	dfResults : xarray.DataArray
		fft results at frequency, f
		
	References
	----------
	https://dsp.stackexchange.com/questions/8611/fft-for-a-single-frequency
	
	Example
	-------
	::
		
		dt=2e-6
		f=1e3
		t=_np.arange(0,10e-3,dt)
		y=_np.sin(2*_np.pi*t*f)
		da=_xr.DataArray( 	y,
							dims=['t'],
							coords={'t':t})
		dfResults=fftSingleFreq(	da,
									f,
									plot=True)

	"""
	#TODO make this function work for simultaneous inputs
	
	# init
	N=da.shape[0]
	
	# fft (complex, amplitude, and phase)
# 	for i,(key,val) in enumerate(df.iteritems()):
#		print(key)
# 	results=_xr.DataArray(	dims=['f'],)
	result=_np.nansum(da.data*_np.exp(-1j*2*_np.pi*f*da.t.data))*2./N
	
	if plot==True:
		
		da_fft=fft(da,plot=True, realAmplitudeUnits=True)
		fig=_plt.gcf()
		ax=_plt.gca()
		ax.set_title( "fft(y) at f = %.3f is %s"%(f, str(result)) )
		
	return result
# 		dfResults.at['fft',key]=fft
# 		amp=_np.abs(fft)
# 		dfResults.at['amp',key]=amp
# 		phase=wrapPhase(_np.arctan2(_np.imag(fft),_np.real(fft))+_np.pi/2)
# 		dfResults.at['phase',key]=phase
# #		print(phase)
# 
# 		# optional plot
# 		if plot==True:
# 			fig,ax=_plt.subplots()
# 			ax.set_title("%s\nFreq: %.1f, Avg. amplitude: %.3f, Avg. phase: %.3f"% (key,f,amp,phase))
# 			ax.plot(val)
# 			ax.plot(val.index,amp*_np.ones(val.shape[0])) 
# 		
# 	return dfResults



# def ifft_df(	dfFFT,
# 				t=None,
# 				plot=False,
# 				invertNormalizedAmplitude=True,
# 				returnRealOnly=True):
# 	
# 	"""
# 	Examples
# 	--------
# 	
# 	Example1::
# 		
# 		import numpy as np
# 		dt=2e-6
# 		f=1e3
# 		t=np.arange(0,10e-3,dt)
# 		y=np.sin(2*np.pi*t*f)
# 		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
# 							index=t,
# 							columns=['a','b','c'])
# 		dfFFT=fft_df(df,plot=False,normalizeAmplitude=False)
# 		
# 		df2=ifft_df(dfFFT)
# 		
# 		for key,val in df.iteritems():
# 			
# 			_plt.figure()
# 			_plt.plot(df.index,val)
# 			sig=df2[key]
# 			sig2=_pd.DataFrame(sig.abs())*np.exp(phase_df(sig))
# 			_plt.plot(df2.index,sig)
# 			
# 			
# 	Example2::
# 			
# 		import pandas as pd
# 		import numpy as np
# 		import johnspythonlibrary2 as jpl2
# 		import matplotlib.pyplot as plt
# 		
# 		# input signal
# 		dt=2e-6
# 		f=1e3
# 		t=np.arange(0,10e-3,dt)
# 		y1=jpl2.Process.SigGen.chirp(t,[1e-3,9e-3],[1e3,1e5])
# 		df1=pd.DataFrame(y1,index=t)
# 		
# 		# filter type 1 : butterworth filter
# 		y2=jpl2.Process.Filters.butterworthFilter(df1,10e3,plot=False)
# 		df2=pd.DataFrame(y2,index=t)
# 		
# 		# filter type 2 : IFFT reconstructed Butterworth filter
# 		tf=jpl2.Process.TF.calcTF(df1,df2,plot=False)
# 		dfFFT=fft_df(df1,plot=False,normalizeAmplitude=False)
# 		df3=ifft_df(pd.DataFrame(tf['lowpassFiltered']*dfFFT[0]))
# 		
# 		# plots
# 		fig,ax=plt.subplots(2,sharex=True)
# 		ax[0].plot(df1,label='Input signal')
# 		ax[0].plot(df2,label='Output signal')
# 		ax[1].plot(df1,label='Input signal')
# 		ax[1].plot(df3,label='Output signal')
# 		_plot.finalizeSubplot( 	ax[0],
# 								subtitle='Butterworth filter')
# 		_plot.finalizeSubplot( 	ax[1],
# 								subtitle='IFFT reconstructed Butterworth filter')
# 		
# 	"""
# 	
# 	if type(dfFFT)==_pd.core.series.Series:
# 		dfFFT=_pd.DataFrame(dfFFT)
# 	
# 	# create a time basis if not provided
# 	if t==None:
# 		N=dfFFT.shape[0]
# 		df=dfFFT.index[1]-dfFFT.index[0]
# 		dt=1/df/N
# 		t=_np.arange(0,N)*dt
# 		
# 	# IFFT function
# 	from numpy.fft import ifft
# 	#from scipy.fftpack.ifft
# 	dfIFFT=dfFFT.apply(ifft,axis=0).set_index(t)
# 	
# 	# option
# 	if returnRealOnly==True:
# 		dfIFFT=_pd.DataFrame(_np.real(dfIFFT),index=dfIFFT.index,columns=dfIFFT.columns)
# 	
# 	# plots
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		ax.plot(dfIFFT)
# 		
# 	return dfIFFT


def ifft(	daFFT,
			t=None,
			plot=False,
			invertNormalizedAmplitude=True,
			returnRealOnly=True):
	
	"""
	Examples
	--------
	
	Example 1::
		
		import numpy as np
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		da=_xr.DataArray( 	y,
							dims=['t'],
							coords={'t':t})
		daFFT=fft(da,plot=False,normalizeAmplitude=False)
		
		df2=ifft(daFFT,t=t,plot=True)
		ax=_plt.gca()
		da.plot(ax=ax,label='original')
		ax.legend()
		
	
	Example 2::
		
		import numpy as np
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		ds=_xr.Dataset( {	'y1':_xr.DataArray(y, 	dims=['t'],	coords={'t':t},	name='y1' ),
							'y2':_xr.DataArray(y*1.2, dims=['t'],	coords={'t':t},	name='y1' ),
							'y3':_xr.DataArray(y*1.4, dims=['t'],	coords={'t':t},	name='y1' )})
		daFFT=fft(ds,plot=False,normalizeAmplitude=False)
		
		df2=ifft(daFFT,t=t,plot=True)
		
			
	Example 3::
			
		import pandas as pd
		import numpy as np
		import johnspythonlibrary2 as jpl2
		import matplotlib.pyplot as plt
		
		# input signal
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y1=jpl2.Process.SigGen.chirp(t,[1e-3,9e-3],[1e3,1e5])
		da1=_xr.DataArray(y1,dims=['t'],	coords={'t':t},	name='y1')
		
		# filter type 1 : butterworth filter
		da2=jpl2.Process.Filters.butterworth(da1,10e3,plot=False)
		
		# filter type 2 : IFFT reconstructed Butterworth filter
		tf=jpl2.Process.TF.calcTF(da1,da2,plot=False)
		daFFT=fft(da1,plot=False,normalizeAmplitude=False)
		da3=ifft(tf*daFFT,t=t)
		
		# plots
		fig,ax=plt.subplots(2,sharex=True)
		ax[0].plot(da1,label='Input signal')
		ax[0].plot(da2,label='Output signal')
		ax[1].plot(da1,label='Input signal')
		ax[1].plot(da3,label='Output signal')
		_plot.finalizeSubplot( 	ax[0],
								subtitle='Butterworth filter')
		_plot.finalizeSubplot( 	ax[1],
								subtitle='IFFT reconstructed Butterworth filter')
		
	"""
	import xarray as xr
	
	# check input
	if type(daFFT) not in [xr.core.dataarray.DataArray,xr.core.dataset.Dataset]:
		raise Exception('Input data not formatted correctly')
	if type(daFFT) in [xr.core.dataarray.Dataset]:
		return daFFT.apply( ifft,
							t=t,
							plot=plot,
							invertNormalizedAmplitude=invertNormalizedAmplitude,
							returnRealOnly=returnRealOnly)
    
	# check if input contains NaN and then remove them
	daFFT[_np.where(_np.isnan(daFFT))] = 0.0 + 0.0*1j
	
	# create a time basis if not provided
	if type(t)==type(None):
		N=daFFT.f.shape[0]
		df=float(daFFT.f[1]-daFFT.f[0])
		dt=1.0/df/N
		t=_np.arange(N)*dt
		
	# IFFT function
	from numpy.fft import ifft as np_ifft
	#from scipy.fftpack import ifft
	daIFFT=_xr.DataArray(	np_ifft(	daFFT),
							dims=['t'],
							coords={'t':t})
	
	# option
	if returnRealOnly==True:
		daIFFT=_np.real(daIFFT)
	
	# plots
	if plot==True:
		fig,ax=_plt.subplots()
		daIFFT.plot(ax=ax)
		
	return daIFFT
	

# def stft_df(	df,
# 				numberSamplesPerSegment=1000,
# 				numberSamplesToOverlap=500,
# 				frequencyResolutionScalingFactor=1.,
# 				plot=False,
# 				verbose=True,
# 				logScale=False):
# 	"""
# 	Short time fourier transform across a range of frequencies
# 	
# 	Parameters
# 	----------
# 	df : pandas.core.frame.DataFrame
# 		1D DataFrame of signal
# 		index = time
# 	numberSamplesPerSegment : int
# 		width of the moving window
# 	numberSamplesToOverlap : int
# 		number of samples to share with the previous window
# 		default is N/2 where N=numberSamplesPerSegment
# 		N-1 is also a good value for detailed analysis but uses a LOT of 
# 		memory and processing power
# 	frequencyResolutionScalingFactor : float
# 		adjust to greater than 1 to increase the number of frequency bins
# 		default is 1.0
# 		2.0, 3.0 , 4.0, etc. are reasonable values
# 	plot : bool
# 		plots results
# 	verbose : bool
# 		prints misc. info related to the frequency limits
# 		
# 	Returns
# 	-------
# 	dfResult : pandas dataframe
# 		index is time. columns is frequency.  values are the complex results at each time and frequency.	
# 	
# 	Examples
# 	--------
# 	Example1::
# 		
# 		# create fake signal.
# 		import numpy as np
# 		import xarray as xr
# 		
# 		fs = 10e3
# 		N = 1e5
# 		amp = 2 * np.sqrt(2)
# 		noise_power = 0.01 * fs / 2
# 		time = np.arange(N) / float(fs)
# 		mod = 500*np.cos(2*np.pi*0.25*time)
# 		carrier = amp * np.sin(2*np.pi*3e3*time + mod)
# 		noise = np.random.normal(scale=np.sqrt(noise_power),
# 		                         size=time.shape)
# 		noise *= np.exp(-time/3)
# 		x = carrier + noise + 1*np.cos(2*np.pi*time*2000)
# 		df = _pd.DataFrame(x,index=time)
# 		
# 		# function call
# 		dfResult=stft(df,plot=True)
# 		
# 		
# 	Example2::
# 		
# 		import numpy as np
# 		import xarray as xr
# 		
# 		# create fake signal.
# 		fs = 10e3
# 		N = 1e5
# 		amp = 2 * np.sqrt(2)
# 		noise_power = 0.01 * fs / 2
# 		time = np.arange(N) / float(fs)
# 		mod = 200*np.cos(2*np.pi*0.25e1*time)
# 		carrier = amp * np.sin(2*np.pi*1e3*time + mod)
# 		noise = np.random.normal(scale=np.sqrt(noise_power),
# 		                         size=time.shape)
# 		noise *= np.exp(-time/5)
# 		x = carrier + noise + 1*np.cos(2*np.pi*time*2000)
# 		df = _pd.DataFrame(x,index=time)
# 		
# 		# function call
# 		dfResult=stft(df,plot=True)
# 		
# 	Notes
# 	-----
# 		1. The sampling rate sets the upper limit on frequency resolution
# 		2. numberSamplesPerSegment sets the lower limit on the (effective) frequency resolution 
# 		3. numberSamplesPerSegment sets an (effective) upper limit on the time responsiveness of the algorithm (meaning, if the frequency rapidly shifts from one value to another)

# 	"""
# 	from scipy.signal import stft as scipystft
# 	import numpy as np
# 	
# 	if type(df)==_pd.core.series.Series:
# 		df=_pd.DataFrame(df)
# 	
# 	dt=df.index[1]-df.index[0]
# 	fs=1./dt
# 	
# 	if verbose:
# 		
# 		print("Sampling rate: %.3e Hz"%fs)
# 		
# 		timeWindow=dt*numberSamplesPerSegment
# 		print("Width of sliding time window: %.3e s"%timeWindow)
# 	
# 		# lowest frequency to get at least one full wavelength
# 		fLow=1./(numberSamplesPerSegment*dt)
# 		print("Lowest freq. to get at least one full wavelength: %.2f" % fLow )
# 	
# 		# frequency upper limit
# 		nyF=fs/2.
# 		print("Nyquist freq. (freq. upperlimit): %.2f" % nyF)
# 	
# 	fOut,tOut,zOut=scipystft(df.iloc[:,0].values,fs,nperseg=numberSamplesPerSegment,noverlap=numberSamplesToOverlap,nfft=df.shape[0]*frequencyResolutionScalingFactor)
# 	zOut*=2 # TODO(John) double check this scaling factor.   Then cite a reason for it.  (I don't like arbitrary factors sitting around)
# 	tOut+=df.index[0]
# 	
# 	dfResult=_pd.DataFrame(zOut.transpose(),index=tOut,columns=fOut)

# 	if plot==True:
# 		
# 		if logScale==False:
# 			fig,ax,cax=_plot.subplotsWithColormaps(1)
# 			levels=np.linspace(0,dfResult.abs().max().max(),61)
# 			pc=ax.contourf(dfResult.index,dfResult.columns,np.abs(dfResult.values.transpose()),levels=levels,cmap='Blues')
# 			fig.colorbar(pc,ax=ax,cax=cax)
# 			_plot.finalizeSubplot(ax,
# 								 xlabel='Time (s)',
# 								 ylabel='Frequency (Hz)',
# 								 legendOn=False)
# 			_plot.finalizeFigure(fig)
# 		else:
# 			raise Exception('Not implemented yet...')
# 			#TODO implement stft spectrogram plot with logscale on the zaxis (colorbar) 
# 			# starting reference : https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/contourf_log.html
# 			
# 			
# 	return dfResult



def stft(	da,
			numberSamplesPerSegment=1000,
			numberSamplesToOverlap=500,
			frequencyResolutionScalingFactor=1.,
			plot=False,
			verbose=True,
			logScale=False,
			window='hann'):
	"""
	Short time fourier transform across a range of frequencies
	
	Parameters
	----------
	da : xarray.DataArray
		1D DataFrame of signal
	numberSamplesPerSegment : int
		width of the moving window
	numberSamplesToOverlap : int
		number of samples to share with the previous window
		default is N/2 where N=numberSamplesPerSegment
		N-1 is also a good value for detailed analysis but uses a LOT of 
		memory and processing power
	frequencyResolutionScalingFactor : float
		adjust to greater than 1 to increase the number of frequency bins
		default is 1.0
		2.0, 3.0 , 4.0, etc. are reasonable values
	plot : bool
		plots results
	verbose : bool
		prints misc. info related to the frequency limits
		
	Returns
	-------
	da_stft : pandas dataframe
		values are the complex results at each time and frequency.	
	
	Examples
	--------
	Example1::
		
		# create fake signal.
		import numpy as np
		import xarray as xr
		
		fs = 10e3
		N = 1e5
		amp = 2 * np.sqrt(2)
		noise_power = 0.01 * fs / 2
		time = np.arange(N) / float(fs)
		mod = 500*np.cos(2*np.pi*0.25*time)
		carrier = amp * np.sin(2*np.pi*3e3*time + mod)
		noise = np.random.normal(scale=np.sqrt(noise_power),
		                         size=time.shape)
		noise *= np.exp(-time/3)
		x = carrier + noise + 1*np.cos(2*np.pi*time*2000)
		da = _xr.DataArray(x,
						 dims=['t'],
						 coords={'t':time})
		
		# function call
		daResult=stft(da,plot=True)
		
		
	Example2::
		
		# create fake signal.
		import numpy as np
		import xarray as xr
		
		fs = 10e3
		N = 1e5
		amp = 2 * np.sqrt(2)
		noise_power = 0.01 * fs / 2
		time = np.arange(N) / float(fs)
		mod = 500*np.cos(2*np.pi*0.25*time)
		carrier = amp * np.sin(2*np.pi*3e3*time + mod)
		noise = np.random.normal(scale=np.sqrt(noise_power),
		                         size=time.shape)
		noise *= np.exp(-time/3)
		x = carrier + noise + 1*np.cos(2*np.pi*time*2000)
		da1 = _xr.DataArray(x,
						 dims=['t'],
						 coords={'t':time})
		da2 = _xr.DataArray(x,
						 dims=['t'],
						 coords={'t':time})
		ds=_xr.Dataset({'da1':da1,
					   'da2':da2})
		
		# function call
		daResult=stft(ds,plot=True)
		
		
	Example3::
		
		import numpy as np
		import xarray as xr
		
		# create fake signal.
		fs = 10e3
		N = 1e5
		amp = 2 * np.sqrt(2)
		noise_power = 0.01 * fs / 2
		time = np.arange(N) / float(fs)
		mod = 200*np.cos(2*np.pi*0.25e1*time)
		carrier = amp * np.sin(2*np.pi*1e3*time + mod)
		noise = np.random.normal(scale=np.sqrt(noise_power),
		                         size=time.shape)
		noise *= np.exp(-time/5)
		x = carrier + noise + 1*np.cos(2*np.pi*time*2000)
		da = _xr.DataArray(x,dims=['t'],coords={'t':time})
		
		# function call
		daResult=stft(da,plot=True)
		
	Notes
	-----
		1. The sampling rate sets the upper limit on frequency resolution
		2. numberSamplesPerSegment sets the lower limit on the (effective) frequency resolution 
		3. numberSamplesPerSegment sets an (effective) upper limit on the time responsiveness of the algorithm (meaning, if the frequency rapidly shifts from one value to another)

	"""
	from scipy.signal import stft as scipystft
	import numpy as np
	import xarray as xr
	
	# check input
	if type(da) not in [xr.core.dataarray.DataArray,xr.core.dataset.Dataset]:
		raise Exception('Input data not formatted correctly')
	if type(da) in [xr.core.dataarray.Dataset]:
		return da.apply( stft,
			numberSamplesPerSegment=numberSamplesPerSegment,
			numberSamplesToOverlap=numberSamplesToOverlap,
			frequencyResolutionScalingFactor=frequencyResolutionScalingFactor,
			plot=plot,
			verbose=verbose,
			logScale=logScale)
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'%(str(da.dims)))
	
	params = signal_spectral_properties(da,nperseg=numberSamplesPerSegment,verbose=verbose)
	fs = params['f_s'] #dt, f_s, f_nyquist, time_window, f_res
	
	fOut,tOut,zOut=scipystft(	da,
								fs,
								nperseg=numberSamplesPerSegment,
								noverlap=numberSamplesToOverlap,
								nfft=da.shape[0]*frequencyResolutionScalingFactor,
								window=window)
	zOut*=2 # TODO(John) double check this scaling factor.   Then cite a reason for it.  (I don't like arbitrary factors sitting around)
	tOut+=da.t[0].data
	
	da_stft=	_xr.DataArray(	zOut,
								dims=['f','t'],
								coords={'t':tOut,'f':fOut})
	da_stft_abs=np.abs(da_stft)
	
	if plot==True:
		_plt.figure()
		if logScale==False:
			da_stft_abs.plot(vmin=da_stft_abs.min(),vmax=da_stft_abs.max())
		else:
			
			from matplotlib.colors import LogNorm
			da_stft_abs.plot(norm=LogNorm(vmin=da_stft_abs.min(),vmax=da_stft_abs.max()))
			
	return da_stft
	
	
# # retire this function?
# def stftSingleFrequency_df(df,
# 						   freq,
# 						   windowSizeInWavelengths=2,
# 						   plot=False,
# 						   verbose=True,):
# 	"""
# 	Short-time Fourier transform of a single frequency.  Uses a Hann window
# 	
# 	Parameters
# 	----------
# 	df : pandas.core.frame.DataFrame
# 		columns contain signals
# 		index = time
# 	f : float
# 		frequency (in Hz) to do the analysis
# 	windowSizeInWavelengths : float
# 		width of the moving stft window, units in wavelengths at frequency, freq
# 	plot : bool
# 		plot results
# 	verbose : bool
# 		print results and related frequency information
# 		
# 	Returns
# 	-------
# 	dfComplex : pandas.core.frame.DataFrame
# 		STFT complex results		
# 	dfAmp : 
# 		STFT amplitude	
# 	dfPhase :
# 		STFT phase	
# 		
# 	References
# 	----------
# 	https://en.wikipedia.org/wiki/Short-time_Fourier_transform
# 	https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
# 	
# 	Example
# 	-------
# 	::
# 		
# 		import numpy as np
# 		import pandas as pd
# 		
# 		dt=2e-6
# 		t=np.arange(0,10e-3,dt)
# 		f=1.5e3
# 		y=np.sin(2*np.pi*t*f+0.25*np.pi)
# 		
# 		N=t.shape[0]
# 		n=np.arange(0,N)
# 		hannWindow=np.sin(np.pi*n/(N-1.))**2
# 		hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
# 		
# 		df=pd.DataFrame(np.array([y,y*1.1,y*1.2]).transpose(),index=t,columns=['a','b','c'])
# 		
# 		dfComplex,dfAmp,dfPhase=stftSingleFrequency_df(df,f,plot=True)
# 	"""
# 	
# 	import numpy as np
# 	
# 	# initial calculations
# 	dt=df.index[1]-df.index[0]
# 	fs=1./dt
# 	N=np.ceil(windowSizeInWavelengths*fs/freq).astype(int)
# 	dN=1
# 	M=df.shape[0]
# 	
# 	# calculate hann window
# 	n=np.arange(0,N)
# 	hannWindow=np.sin(np.pi*n/(N-1.))**2
# 	hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
# 	dfHann=_pd.DataFrame([hannWindow]*df.shape[1]).transpose()
# 			
# 	# calculate steps
# 	steps=np.arange(0,M-N,dN)
# 		
# 	if verbose:
# 		# time window
# 		timeWindow=dt*N 
# 		print("Time window: %.3e s"%timeWindow)
# 	
# 		# lowest frequency to get a full wavelength
# 		fLow=1./(N*dt)
# 		print("Lowest freq. to get 1 full wavelength, %.2f" % fLow )
# 	
# 		# highest frequency
# 		nyF=fs/2.
# 		print("Nyquist freq., %.2f" % nyF)
# 	
# 	# initialize arrays
# 	dfAmp=_pd.DataFrame(index=df.index[steps+int(N/2)],
# 							columns=df.columns,
# 							dtype=complex)
# 	dfComplex=_pd.DataFrame(index=df.index[steps+int(N/2)],
# 							columns=df.columns,
# 							dtype=complex)
# 	dfPhase=_pd.DataFrame(index=df.index[steps+int(N/2)],
# 							columns=df.columns,
# 							dtype=float)
# 	
# 	# perform analysis
# 	#TODO optomize this section of code.  should be a convolution function somewhere
# 	for i in range(0,len(steps)):
# 		temp=_pd.DataFrame(df.iloc[steps[i]:steps[i]+N].values,
# 					index=dt*(n-n.mean()),
# 					columns=df.columns)
# 		temp=_xr.DataArray(temp.values, dims='t',coords=[steps+int(N/2)])
# 		dfOut=fftSingleFreq(temp*dfHann.values,freq,plot=False)
# 			
# 		dfComplex.iloc[i,:]=dfOut.loc['fft']
# 		dfAmp.iloc[i,:]=dfOut.loc['amp']
# 		dfPhase.iloc[i,:]=dfOut.loc['phase']
# 		
# 	if plot==True:
# 		for i,(key,val) in enumerate(df.iteritems()):
# 			
# 			fig,(ax1,ax2)=_plt.subplots(2,sharex=True)
# 			ax1.plot(val)
# 			ax1.plot(dfAmp[key])
# 			ax2.plot(dfPhase[key],'.')
# 		
# 	return dfComplex,dfAmp,dfPhase



# retire this function?
def stftSingleFrequency(da,
						   freq,
						   windowSizeInWavelengths=2,
						   plot=False,
						   verbose=True,):
	"""
	Short-time Fourier transform of a single frequency.  Uses a Hann window
	
	Parameters
	----------
	da : xarray dataarray
		data
	f : float
		frequency (in Hz) to do the analysis
	windowSizeInWavelengths : float
		width of the moving stft window, units in wavelengths at frequency, freq
	plot : bool
		plot results
	verbose : bool
		print results and related frequency information
		
	Returns
	-------
	dfComplex : pandas.core.frame.DataFrame
		STFT complex results		
	dfAmp : 
		STFT amplitude	
	dfPhase :
		STFT phase	
		
	References
	----------
	https://en.wikipedia.org/wiki/Short-time_Fourier_transform
	https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
	
	Example
	-------
	::
		
		import numpy as np
		import pandas as pd
		import xarray as xr
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		y=np.sin(2*np.pi*t*f+0.25*np.pi)
		
		N=t.shape[0]
		n=np.arange(0,N)
		hannWindow=np.sin(np.pi*n/(N-1.))**2
		hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
		
		da=xr.DataArray(y, dims='t', coords=[t])
		
		da_complex=stftSingleFrequency(da,f,plot=True)
	"""
	
	import numpy as np
	
	# initial calculations
	dt=da.t[1].data-da.t[0].data
	fs=1./dt
	N=np.ceil(windowSizeInWavelengths*fs/freq).astype(int)
	dN=1
	M=len(da)
	
	# calculate hann window
	n=np.arange(0,N)
	hannWindow=np.sin(np.pi*n/(N-1.))**2
	hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
# 	dfHann=_pd.DataFrame(hannWindow.transpose()
			
	# calculate steps
	steps=np.arange(0,M-N,dN)
		
	if verbose:
		# time window
		timeWindow=dt*N 
		print("Time window: %.3e s"%timeWindow)
	
		# lowest frequency to get a full wavelength
		fLow=1./(N*dt)
		print("Lowest freq. to get 1 full wavelength, %.2f" % fLow )
	
		# highest frequency
		nyF=fs/2.
		print("Nyquist freq., %.2f" % nyF)
	
	# initialize arrays
# 	dfComplex=[]
	step_time=da.t.data[steps+N//2]
	da_complex=_xr.DataArray(np.zeros(len(step_time),dtype=complex), dims='t', coords=[step_time])
	
	# perform analysis
	#TODO optomize this section of code.  should be a convolution function somewhere
	for i in range(0,len(steps)):
		temp=da[steps[i]:steps[i]+N].data
		temp_time=da[steps[i]:steps[i]+N].t.data
		dfOut=fftSingleFreq( _xr.DataArray(temp*hannWindow, dims='t', coords=[temp_time]),freq,plot=False)
			
		da_complex[i]=dfOut
# 		dfComplex.append(dfOut)
	
# 	da_complex = _xr.DataArray( np.array(dfComplex), dims='t', coords=[da.t.data[steps+N//2]])
		
	if plot==True:
		fig,(ax1,ax2)=_plt.subplots(2,sharex=True)
		da_complex.real.plot(ax=ax1)
		da_complex.imag.plot(ax=ax2, linestyle='', marker='.')
	
		
	return da_complex



	
###############################################################################
#%% Coherence

	
def _coherenceComplex_old(	x, 
							y, 
							fs=1.0, 
							window='hann', 
							nperseg=None, 
							noverlap=None,
							nfft=None, 
							detrend='constant', 
							axis=-1):
	r"""
	This is a blatant copy of scipy.signal.coherence with one modification.  
	This modification has the code return the complex coherence instead of the 
	absolute value.  

	Parameters
	----------
	x : array_like
		Time series of measurement values
	y : array_like
		Time series of measurement values
	fs : float, optional
		Sampling frequency of the `x` and `y` time series. Defaults
		to 1.0.
	window : str or tuple or array_like, optional
		Desired window to use. If `window` is a string or tuple, it is
		passed to `get_window` to generate the window values, which are
		DFT-even by default. See `get_window` for a list of windows and
		required parameters. If `window` is array_like it will be used
		directly as the window and its length must be nperseg. Defaults
		to a Hann window.
	nperseg : int, optional
		Length of each segment. Defaults to None, but if window is str or
		tuple, is set to 256, and if window is array_like, is set to the
		length of the window.
	noverlap: int, optional
		Number of points to overlap between segments. If `None`,
		``noverlap = nperseg // 2``. Defaults to `None`.
	nfft : int, optional
		Length of the FFT used, if a zero padded FFT is desired. If
		`None`, the FFT length is `nperseg`. Defaults to `None`.
	detrend : str or function or `False`, optional
		Specifies how to detrend each segment. If `detrend` is a
		string, it is passed as the `type` argument to the `detrend`
		function. If it is a function, it takes a segment and returns a
		detrended segment. If `detrend` is `False`, no detrending is
		done. Defaults to 'constant'.
	axis : int, optional
		Axis along which the coherence is computed for both inputs; the
		default is over the last axis (i.e. ``axis=-1``).

	Returns
	-------
	f : ndarray
		Array of sample frequencies.
	Cxy : ndarray
		Magnitude squared coherence of x and y.

	See Also
	--------
	periodogram: Simple, optionally modified periodogram
	lombscargle: Lomb-Scargle periodogram for unevenly sampled data
	welch: Power spectral density by Welch's method.
	csd: Cross spectral density by Welch's method.

	Notes
	--------
	An appropriate amount of overlap will depend on the choice of window
	and on your requirements. For the default Hann window an overlap of
	50% is a reasonable trade off between accurately estimating the
	signal power, while not over counting any of the data. Narrower
	windows may require a larger overlap.

	.. versionadded:: 0.16.0

	References
	----------
	.. [1] P. Welch, "The use of the fast Fourier transform for the
		   estimation of power spectra: A method based on time averaging
		   over short, modified periodograms", IEEE Trans. Audio
		   Electroacoust. vol. 15, pp. 70-73, 1967.
	.. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
		   Signals" Prentice Hall, 2005

	Examples
	--------
	::
		
		from scipy import signal
		import matplotlib.pyplot as plt
		import numpy as np
	
		#Generate two test signals with some common features.
	
		fs = 10e3
		N = 1e5
		amp = 20
		freq = 1234.0
		noise_power = 0.001 * fs / 2
		time = np.arange(N) / fs
		b, a = signal.butter(2, 0.25, 'low')
		x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
		y = signal.lfilter(b, a, x)
		x += amp*np.sin(2*np.pi*freq*time+np.pi)
		y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
	
		#Compute and plot the coherence.
	
		f, Cxy = _coherenceComplex(x, y, fs, nperseg=1024)
		plt.semilogy(f, np.abs(Cxy))
		plt.xlabel('frequency [Hz]')
		plt.ylabel('Coherence')
		plt.show()
	"""
	
	import scipy.signal as signal
	import numpy as np
	freqs, Pxx = signal.welch(x, fs=fs, window=window, nperseg=nperseg,
					   noverlap=noverlap, nfft=nfft, detrend=detrend,
					   axis=axis)
	_, Pyy = signal.welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
				   nfft=nfft, detrend=detrend, axis=axis)
	_, Pxy = signal.csd(x, y, fs=fs, window=window, nperseg=nperseg,
				 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

	Cxy = Pxy / np.sqrt(Pxx *Pyy) 
	Cxy *= Cxy

	return freqs, Cxy


def _coherenceComplex(da1,da2, window='hann', nperseg=None, noverlap=None,
			  nfft=None, detrend='constant', axis=-1):
	r"""
	This is a blatant copy of scipy.linspace.coherence with one modification.  
	This modification has the code return the complex coherence instead of the 
	absolute value.  

	Parameters
	----------
	x : array_like
		Time series of measurement values
	y : array_like
		Time series of measurement values
	fs : float, optional
		Sampling frequency of the `x` and `y` time series. Defaults
		to 1.0.
	window : str or tuple or array_like, optional
		Desired window to use. If `window` is a string or tuple, it is
		passed to `get_window` to generate the window values, which are
		DFT-even by default. See `get_window` for a list of windows and
		required parameters. If `window` is array_like it will be used
		directly as the window and its length must be nperseg. Defaults
		to a Hann window.
	nperseg : int, optional
		Length of each segment. Defaults to None, but if window is str or
		tuple, is set to 256, and if window is array_like, is set to the
		length of the window.
	noverlap: int, optional
		Number of points to overlap between segments. If `None`,
		``noverlap = nperseg // 2``. Defaults to `None`.
	nfft : int, optional
		Length of the FFT used, if a zero padded FFT is desired. If
		`None`, the FFT length is `nperseg`. Defaults to `None`.
	detrend : str or function or `False`, optional
		Specifies how to detrend each segment. If `detrend` is a
		string, it is passed as the `type` argument to the `detrend`
		function. If it is a function, it takes a segment and returns a
		detrended segment. If `detrend` is `False`, no detrending is
		done. Defaults to 'constant'.
	axis : int, optional
		Axis along which the coherence is computed for both inputs; the
		default is over the last axis (i.e. ``axis=-1``).

	Returns
	-------
	f : ndarray
		Array of sample frequencies.
	Cxy : ndarray
		Magnitude squared coherence of x and y.

	See Also
	--------
	periodogram: Simple, optionally modified periodogram
	lombscargle: Lomb-Scargle periodogram for unevenly sampled data
	welch: Power spectral density by Welch's method.
	csd: Cross spectral density by Welch's method.

	Notes
	--------
	An appropriate amount of overlap will depend on the choice of window
	and on your requirements. For the default Hann window an overlap of
	50% is a reasonable trade off between accurately estimating the
	signal power, while not over counting any of the data. Narrower
	windows may require a larger overlap.

	.. versionadded:: 0.16.0

	References
	----------
	.. [1] P. Welch, "The use of the fast Fourier transform for the
		   estimation of power spectra: A method based on time averaging
		   over short, modified periodograms", IEEE Trans. Audio
		   Electroacoust. vol. 15, pp. 70-73, 1967.
	.. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
		   Signals" Prentice Hall, 2005

	Examples
	--------
	Example 1::
		
		from scipy import signal
		import matplotlib.pyplot as plt
		import numpy as np
	
		#Generate two test signals with some common features.
	
		fs = 10e3
		N = 1e5
		amp = 20
		freq = 1234.0
		noise_power = 0.001 * fs / 2
		time = np.arange(N) / fs
		b, a = signal.butter(2, 0.25, 'low')
		x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
		y = signal.lfilter(b, a, x)
		x += amp*np.sin(2*np.pi*freq*time+np.pi)
		y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
		da1=_xr.DataArray(x,
							dims=['t'],
							coords={'t':time})
		da2=_xr.DataArray(y,
							dims=['t'],
							coords={'t':time})
	
		#Compute and plot the coherence.
	
		coh = _coherenceComplex(da1,da2, nperseg=1024)
		fig,ax=plt.subplots()
		np.abs(coh).plot(ax=ax,yscale='log')
		plt.xlabel('frequency [Hz]')
		plt.ylabel('Coherence')
		plt.show()
	"""
	
	import scipy.signal as signal
	import numpy as np
	dt=float(da1.t[1]-da1.t[0])
	fs=1.0/dt
	freqs, Pxx = signal.welch(da1, fs=fs, window=window, nperseg=nperseg,
					   noverlap=noverlap, nfft=nfft, detrend=detrend,
					   axis=axis)
	_, Pyy = signal.welch(da2, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
				   nfft=nfft, detrend=detrend, axis=axis)
	_, Pxy = signal.csd(da1, da2, fs=fs, window=window, nperseg=nperseg,
				 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

	Cxy = Pxy / np.sqrt(Pxx *Pyy)   #scipy.signal.coherence code : Cxy = np.abs(Pxy)**2 / Pxx / Pyy
	Cxy *= Cxy

	return _xr.DataArray(Cxy,
						  dims=['f'],
						  coords={'f':freqs})


def coherenceAnalysis(	da1,
						da2,
						numPointsPerSegment=1024,
						plot=False,
						noverlap=None,
						verbose=True,
						removeOffsetAndNormalize=False):
	"""
	Coherence analysis between two signals. Wrapper for scipy's coherence analysis
	
	Parameters
	----------
	da1 : xarray.DataArray
		time dependent data signal 1
	da2 : xarray.DataArray
		time dependent data signal 1
	numPointsPerSegment : int
		default 1024. number of points used in the moving analysis
	plot : bool
		True = plots the results
	noverlap : int
		default None. number of points-overlapped in the moving analysis
	verbose : bool
		If True, plots misc parameters.
		
	Returns
	-------
	coh : xarray.DataArray
		coherence analysis
	
	References
	----------
	* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
	
	Example 1
	---------
	::
		
		## Case where signals have the same phase (with a different offset) and one signal has a lot of noise
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		freq=5
		time = np.arange(0, 10, 0.01)
		phase_signal_1=time*np.pi*2*freq
		phase_signal_2=time*np.pi*2*freq+0.5
		np.random.seed(0)
		da1 = _xr.DataArray(	np.sin(phase_signal_1)+ 1.5*np.random.randn(len(time)),
									dims=['t'],
									coords={'t':time},
									name='signal1'	)*2+1
		da1.attrs=	{'long_name':'signal1',
						'units':'au'}
		da2 = _xr.DataArray(	np.sin(phase_signal_2)+ 0.0*np.random.randn(len(time)),
									dims=['t'],
									coords={'t':time},
									name='signal2'	)
		da2.attrs=	{'long_name':'signal2',
						'units':'au'}
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=0)
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=64)
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=127)


	Example 2
	---------
	::
		
		## Case where signals have the same phase (with a different offset), one signal has two frequency components and noise
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		freq=5
		time = np.arange(0, 10, 0.01)
		phase_signal_1=time*np.pi*2*freq
		phase_signal_2=time*np.pi*2*freq+0.5
		np.random.seed(0)
		da1 = _xr.DataArray(	np.sin(phase_signal_1),
									dims=['t'],
									coords={'t':time},
									name='signal1'	)
		da2 = _xr.DataArray(	np.sin(phase_signal_2) + 2*np.sin(time*np.pi*2*17.123) + 1.20*np.random.randn(len(time)),
									dims=['t'],
									coords={'t':time},
									name='signal2'	)
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=0)
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=64)
		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=127)

	"""
	
	import matplotlib.pyplot as plt
	import numpy as np
	
	if 'time' in da1.dims:
		da1=da1.rename({'time':'t'})
	if 'time' in da2.dims:
		da2=da2.rename({'time':'t'})
	if 't' not in da1.dims and 't' not in da2.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'%(str(da1.dims)+str(da2.dims)))
		
	if removeOffsetAndNormalize == True:
		da1=(da1.copy()-da1.mean())/da1.std()
		da2=(da2.copy()-da2.mean())/da2.std()
	
	dt,fs,_,_,f_min=signal_spectral_properties(da1,nperseg=numPointsPerSegment,verbose=verbose).values()
	
	
	# coherence analysis 
	coh = _coherenceComplex(da1,da2, nperseg=numPointsPerSegment,noverlap=noverlap,nfft=numPointsPerSegment*2)
	
	# optional plot
	if plot==True:
		fig,(ax1A,ax2,ax3)=plt.subplots(nrows=3)
		ax1B=ax1A.twinx()
		da1.plot(ax=ax1A,label=da1.name)
		da2.plot(ax=ax1B,label=da2.name,color='tab:blue')
		ax1A.legend(loc='upper left')
		ax1B.legend(loc='upper right')
		ax1A.set_xlabel('time (s)')
		ax1A.set_ylabel(da1.name)
		ax1B.set_ylabel(da2.name)
		
		f1=fft_average((da1-da1.mean())/da1.std(),nperseg=numPointsPerSegment,verbose=False)
		np.abs(f1).where(f1.f>=f_min).plot(ax=ax2,label=da1.name,yscale='log')
		f2=fft_average((da2-da2.mean())/da2.std(),nperseg=numPointsPerSegment,verbose=False)
		np.abs(f2).where(f2.f>=f_min).plot(ax=ax2,label=da2.name,yscale='log')
		ax2.legend()
		
		np.abs(coh).plot(ax=ax3,label='Coherence',marker='',linestyle='-')
		ax3.set_xlabel('frequency (Hz)')
		ax3.set_ylabel('Coherence')
		ax3.legend()
		ax3.set_ylim([0,1])
		
		if noverlap == None:
			noverlap=0
		ax1A.set_title('points overlapped = %d of %d'%(noverlap,numPointsPerSegment))
		_finalizeFigure(fig)
		
	return coh


def coherenceAnalysis_old(	t,
							y1,
							y2,
							numPointsPerSegment=1024,
							plot=False,
							noverlap=None,
							verbose=True,
							s1Label='Signal 1',
							s2Label='Signal 2'):
	"""
	Coherence analysis between two signals. Wrapper for scipy's coherence analysis
	
	Parameters
	----------
	t : numpy.array
		time
	y1 : numpy.array
		time dependent data signal 1
	y2 : numpy.array
		time dependent data signal 2
	numPointsPerSegment : int
		default 1024. number of points used in the moving analysis
	plot : bool
		True = plots the results
	TODO(John) : finish remaining parameters
		
	Returns
	-------
	f : numpy.array
		frequencies (in Hz)
	Cxy : numpy.array
		coherence analysis
	
	References
	----------
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
	
	Example 1
	---------
	::
		
		## Case where signals have the same phase (with a different offset) and one signal has a lot of noise
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		freq=5
		time = np.arange(0, 10, 0.01)
		phase_signal_1=time*np.pi*2*freq
		phase_signal_2=time*np.pi*2*freq+0.5
		np.random.seed(0)
		sinewave1 = np.sin(phase_signal_1)+ 1.5*np.random.randn(len(time)) # add white noise to the signal
		sinewave2 = np.sin(phase_signal_2)+ 0.0*np.random.randn(len(time)) # add white noise to the signal
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=0)
		_plt.gcf().get_axes()[0].set_title('points overlapped = 0 of 128')
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=64)
		_plt.gcf().get_axes()[0].set_title('points overlapped = 64 of 128')
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=127)
		_plt.gcf().get_axes()[0].set_title('points overlapped = 127 of 128')
		
	Example 2
	---------
	::
		
		## Case where signals have the same phase (with a different offset), one signal has two frequency components and noise
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		freq=5
		time = np.arange(0, 10, 0.01)
		phase_signal_1=time*np.pi*2*freq
		phase_signal_2=time*np.pi*2*freq+0.5
		np.random.seed(0)
		sinewave1 = np.sin(phase_signal_1)
		sinewave2 = np.sin(phase_signal_2) + 2*np.sin(time*np.pi*2*17.123) + 1.20*np.random.randn(len(time)) # add white noise to the signal
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=0)
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=64)
		f,Cxy=coherenceAnalysis(time,sinewave1,sinewave2,plot=True,numPointsPerSegment=128,noverlap=127)

	"""
	
	import matplotlib.pyplot as plt
	import numpy as np
	
	
	# sampling rate
	dt=t[1]-t[0]
	fs=1./dt
	timeWindow=dt*numPointsPerSegment
	if verbose: print("Time window: %.3e s"%timeWindow)
	
	# lowest frequency to get a full wavelength
	fLow=1./(numPointsPerSegment*dt)
	if verbose: print("Lowest freq. to get 1 wavelength, %.2f" % fLow )
	
	# highest frequency
	nyF=fs/2.
	if verbose: print("Nyquist freq., %.2f" % nyF)
	
	# coherence analysis 
	f, Cxy = _coherenceComplex(y1,y2, fs, nperseg=numPointsPerSegment,noverlap=noverlap,nfft=numPointsPerSegment*2)
	
	# optional plot
	if plot==True:
		fig,(ax1,ax2)=plt.subplots(nrows=2)
		ax1.plot(t,y1,label=s1Label)
		ax1.plot(t,y2,label=s2Label)
		ax1.legend()
		ax1.set_xlabel('time (s)')
		ax2.plot(f, np.abs(Cxy),'.-',label='Coherence')
		# ax2.semilogy(f, Cxy,label='Coherence')
		ax2.set_xlabel('frequency (Hz)')
		ax2.set_ylabel('Coherence')
		ax2.legend()
		ax2.set_ylim([0,1])
		if noverlap == None:
			noverlap=0
		ax1.set_title('points overlapped = %d of %d'%(noverlap,numPointsPerSegment))
		
	return f, Cxy


###############################################################################
#%% Phase related

	


def wrapPhase(phases,center=0):
	""" 
	simple wrap phase function from -pi to +pi.  units in radians
	
	Parameters
	----------
	phases : numpy.ndarray or pandas.core.frame.DataFrame
		array of phase data.  units in radians
		
	Returns
	-------
	wrapped phase (units in radians)
	
	Example
	-------
	Example 1::
		
		phi=_np.arange(-10,10,0.1)
		fig,ax=_plt.subplots()
		ax.plot(phi,phi,'.',label='unwrapped')
		ax.plot(phi,wrapPhase(phi),'x',label='wrapped about phi=0')
		ax.plot(phi,wrapPhase(phi,_np.pi),'+',label='wrapped about phi=pi')
		ax.plot(phi,wrapPhase(phi,-_np.pi*2),'v',label='wrapped about phi=-2pi')
		ax.legend()
		_plt.show()
		
	References
	----------
	  *  https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
 	"""
	# return (phases + _np.pi) % (2 * _np.pi) - _np.pi
	return (phases + _np.pi - center) % (2 * _np.pi) - _np.pi + center
		
	
def unwrapPhase(inData):
	"""
	Takes in phase array (in radians).  I think it needs to be centered about 0.
	Unwraps phase data so that it is continuous.
	This is important for phase data when you want to take it's derivative to
	get frequency.  
	
	Parameters
	----------
	data : numpy.ndarray
		data being unwrapped
		
	Return
	------
	outData : numpy.ndarray
		unwrapped data array
		
	"""
	return _np.unwrap(inData)


def calcPhaseDifference_df(dfX1,dfX2,plot=False,title=''):
	"""
	Calculates the phase difference between two complex signals.
	Also calculates the average and standard deviation of the phase difference.

	Parameters
	----------
	dfX1 : pandas.core.frame.DataFrame
		Complex signal in sine and cosine basis
	dfX2 : pandas.core.frame.DataFrame
		Complex signal in sine and cosine basis
	plot : bool, optional
		Create plot of results. The default is False.

	Returns
	-------
	dfPD : pandas.core.frame.DataFrame
		Phase difference between X1 and X2
		
	References
	----------
	https://vicrucann.github.io/tutorials/phase-average/
	
	Example
	-------
	::
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		y1=np.sin(2*np.pi*t*f+0.25*np.pi)
		y2=np.sin(2*np.pi*t*f+0.75*np.pi)
		
		df1=_pd.DataFrame(y1,index=t)
		df2=_pd.DataFrame(y2,index=t)
		
		dfX1,dfAmp1,dfPhase1=hilbertTransform_df(df1,plot=False)
		dfX2,dfAmp2,dfPhase2=hilbertTransform_df(df2,plot=False)
		
		dfPD,avePhaseDiff,stdPhaseDiff=calcPhaseDifference(dfX1,dfX2,plot=True,)
	"""
	
	# phase difference calc
	S12=dfX1*_np.conj(dfX2)
	dfPD=_pd.DataFrame(_np.arctan2(_np.imag(S12),_np.real(S12)),
							index=dfX1.index)
	
	# phase diff average and standard deviation calc
	X=_np.cos(dfPD)
	Y=_np.sin(dfPD)
	avePhaseDiff=_np.arctan2(Y.mean(),X.mean())
	
	stdPhaseDiff=wrapPhase(dfPD-avePhaseDiff).std()
		
	if plot==True:
		fig,ax=_plt.subplots(3,sharex=True)
		ax[0].plot(dfX1.index,_np.real(dfX1),label='Real')
		ax[0].plot(dfX1.index,_np.imag(dfX1),label='Imag')
		ax[1].plot(dfX2.index,_np.real(dfX2),label='Real')
		ax[1].plot(dfX2.index,_np.imag(dfX2),label='Imag')
		p1=_np.arctan2(_np.imag(dfX1),_np.real(dfX1)) # TODO swap to np.angle instead of arctan2
		p2=_np.arctan2(_np.imag(dfX2),_np.real(dfX2))# TODO swap to np.angle instead of arctan2
		markersize=2
		ax[2].plot(dfX1.index,p1,'.',label='X1 phase',markersize=markersize)
		ax[2].plot(dfX1.index,p2,'.',label='X2 phase',markersize=markersize)
		ax[2].plot(dfX1.index,dfPD,'.',label='Phase diff.',markersize=markersize)
		
		_plot.finalizeSubplot(	ax[0],
								title=r'Ave. Phase Diff = %.2f $\pm$ %.2f rad'%(avePhaseDiff,stdPhaseDiff),
								subtitle='X1')
		_plot.finalizeSubplot(	ax[1],
								subtitle='X2',
								)
		_plot.finalizeSubplot(	ax[2], xlabel='Time',
								ylabel='Rad.',
								yticks=[-_np.pi,0,_np.pi],
								ylim=[-_np.pi,_np.pi],
								subtitle='Phase',
								ytickLabels=[r'$-\pi$','0','$\pi$'])
		_plot.legendOutside(ax[0])
		_plot.legendOutside(ax[1])
		_plot.legendOutside(ax[2])
		_plot.finalizeFigure(fig,figSize=[6,4])
		
	return dfPD,avePhaseDiff,stdPhaseDiff


def calcPhaseDifference(X1,X2,plot=False,title=''):
	"""
	Calculates the phase difference between two complex signals.
	Also calculates the average and standard deviation of the phase difference.

	Parameters
	----------
	X1 : xarray.Dataarray
		Complex signal in sine and cosine basis
	X2 : xarray.Dataarray
		Complex signal in sine and cosine basis
	plot : bool, optional
		Create plot of results. The default is False.

	Returns
	-------
	PD : xarray.Dataarray
		Phase difference between X1 and X2
		
	References
	----------
	https://vicrucann.github.io/tutorials/phase-average/
	
	Example
	-------
	Example 1::
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		from johnspythonlibrary2.Process.SigGen import gaussianNoise
		
		y1=_xr.DataArray( 	np.sin(2*np.pi*t*f+0.25*np.pi)+0.1*gaussianNoise(t),
						   dims=['t'],
						   coords={'t':t})
		y2=_xr.DataArray( np.sin(2*np.pi*t*f+0.75*np.pi)+0.1*gaussianNoise(t),
						   dims=['t'],
						   coords={'t':t})
		
		
		X1,Amp1,Phase1=hilbertTransform(y1,plot=False)
		X2,Amp2,Phase2=hilbertTransform(y2,plot=False)
		
		dfPD,avePhaseDiff,stdPhaseDiff=calcPhaseDifference(X1,X2,plot=True,)
		
	"""
	
	# phase difference calc
	S12=X1*_np.conj(X2)
	PD=_xr.DataArray(_np.arctan2(_np.imag(S12),_np.real(S12))) # TODO swap to np.angle instead of arctan2
	
	# phase diff average and standard deviation calc.  this code converts polar to cartesian coordinates to do the following calculations
	X=_np.cos(PD)
	Y=_np.sin(PD)
	avePhaseDiff=_np.arctan2(Y.mean(),X.mean())
	stdPhaseDiff=wrapPhase(PD-avePhaseDiff).std()
		
	if plot==True:
		fig,ax=_plt.subplots(3,sharex=True)
		_np.real(X1).plot(ax=ax[0],label='Real')
		_np.imag(X1).plot(ax=ax[0],label='Imag')
		_np.real(X2).plot(ax=ax[1],label='Real')
		_np.imag(X2).plot(ax=ax[1],label='Imag')
		p1=_np.arctan2(_np.imag(X1),_np.real(X1)) # TODO swap to np.angle instead of arctan2
		p2=_np.arctan2(_np.imag(X2),_np.real(X2)) # TODO swap to np.angle instead of arctan2
		markersize=2
		p1.plot(ax=ax[2],linestyle='',marker='.',label='X1 phase',markersize=markersize)
		p2.plot(ax=ax[2],linestyle='',marker='.',label='X2 phase',markersize=markersize)
		PD.plot(ax=ax[2],linestyle='',marker='.',label='Phase diff.',markersize=markersize)
		
		_plot.finalizeSubplot(	ax[0],
								title=r'Ave. Phase Diff = %.2f $\pm$ %.2f rad'%(avePhaseDiff,stdPhaseDiff),
								subtitle='X1')
		_plot.finalizeSubplot(	ax[1],
								subtitle='X2')
		_plot.finalizeSubplot(	ax[2], xlabel='Time',
								ylabel='Rad.',
								yticks=[-_np.pi,0,_np.pi],
								ylim=[-_np.pi,_np.pi],
								subtitle='Phase',
								ytickLabels=[r'$-\pi$','0','$\pi$'])
		_plot.legendOutside(ax[0])
		_plot.legendOutside(ax[1])
		_plot.legendOutside(ax[2])
		_plot.finalizeFigure(fig,figSize=[6,4])
		
	return PD,avePhaseDiff,stdPhaseDiff

	
def phase_df(df):
	"""
	Calculates the phase of complex data series.
	Basis can be time, frequency, etc.
	"""
	
	if type(df)==_pd.core.series.Series:
		df=_pd.DataFrame(df)
	
	return _pd.DataFrame(_np.arctan2(_np.imag(df),_np.real(df)), # TODO swap to np.angle instead of arctan2
						      index=df.index,
							  columns=df.columns)
	


###############################################################################
#%% Hilbert



def hilbertTransform_df(df,plot=False):
	""" 
	Hilbert transform.  Shifts signal by 90 deg. and finds the phase between 
	the imaginary and real components
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		each column is data
		index = time		
	plot : bool
		optional plot
		
	Returns
	-------
	dfHilbert : pandas.core.frame.DataFrame
		hilbert transformed signals
	dfAmp : pandas.core.frame.DataFrame
		amplitude of hilbert transformed signals
	dfPhase : pandas.core.frame.DataFrame
		phase of hilbert transformed signals
		
	Example
	-------
	::
		
		
		import numpy as np
		import pandas as pd
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		y=np.sin(2*np.pi*t*f+0.25*np.pi)
		
		df=pd.DataFrame(np.array([y,y*1.1,y*1.2]).transpose(),index=t,columns=['a','b','c'])
		
		dfHilbert,dfAmp,dfPhase=hilbertTransform_df(df,plot=True)
	"""
	from scipy import signal as sig
	
	dfHilbert=_pd.DataFrame(sig.hilbert(df.copy()-df.mean(),axis=0),
						index=df.index,
						columns=df.columns)
	dfAmp=_np.abs(dfHilbert)
	dfPhase=_pd.DataFrame(_np.arctan2(_np.imag(dfHilbert),_np.real(dfHilbert)), # TODO swap to np.angle instead of arctan2
					index=df.index,
					columns=df.columns)
	
	if plot==True:
		dt=df.index[1]-df.index[0]
		freq=_np.gradient(_np.unwrap(dfPhase,axis=0),dt,axis=0)/(2*_np.pi)
		
		for i,(key,val) in enumerate(dfHilbert.iteritems()):
			fig,ax=_plt.subplots(3,sharex=True)
			ax[0].plot(val.index,_np.real(val),label='original')
			ax[0].plot(val.index,_np.imag(val),label=r'90$^o$ shifted')
			ax[0].plot(dfAmp.index,dfAmp[key],label=r'amplitude')
			ax[1].plot(dfPhase.index,dfPhase[key],'.',label='phase')
			ax[2].plot(dfPhase.index,freq[:,i],label='freq')

			_plot.finalizeSubplot(ax[0],
									ylabel='Amplitude',
									subtitle='Signals',
									legendLoc='lower right')
			_plot.finalizeSubplot(ax[1],
									ylabel='Rad.',
									subtitle='Phase',
									legendOn=False,
									yticks=[-_np.pi,-_np.pi/2,0,_np.pi/2,_np.pi],
									ytickLabels=[r'-$\pi$',r'-$\pi$/2',r'0',r'$\pi$/2',r'$\pi$'],
									ylim=[-_np.pi,_np.pi],
									xlabel='Time',)
			_plot.finalizeSubplot(ax[2],
									ylabel='Hz',
									subtitle='Frequency',
									xlabel='Time',
									legendOn=False)
		
	return dfHilbert,dfAmp,dfPhase



def hilbertTransform(da,plot=False):
	""" 
	Hilbert transform.  Shifts signal by 90 deg. and finds the phase between 
	the imaginary and real components
	
	Parameters
	----------
	da : xarray.DataArray
		input signal
	plot : bool
		optional plot
		
	Returns
	-------
	daHilbert : pandas.core.frame.DataFrame
		hilbert transformed signals
	daAmp : pandas.core.frame.DataFrame
		amplitude of hilbert transformed signals
	daPhase : pandas.core.frame.DataFrame
		phase of hilbert transformed signals
		
	Example
	-------
	Example 1::
		
		import numpy as np
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		from johnspythonlibrary2.Process.SigGen import gaussianNoise
		y=np.sin(2*np.pi*t*f+0.25*np.pi) + 0.001*gaussianNoise(t)
		
		da=_xr.DataArray(	y,
							dims=['t'],
							coords={'t':t})
		
		daHilbert,daAmp,daPhase=hilbertTransform(da,plot=True)
		
	"""
	from scipy import signal as sig
	
	if type(da) != _xr.core.dataarray.DataArray:
		raise Exception('Input not formatted correctly')
	if 't' not in da.dims:
		raise Exception('Time coordinate not formatted correctly')
	
	daHilbert=_xr.DataArray(sig.hilbert(da.copy()-da.mean()),
						dims=['t'],
						coords={'t':da.t})
	daAmp=_np.abs(daHilbert)
	daPhase=_xr.DataArray(	_np.arctan2(_np.imag(daHilbert),_np.real(daHilbert))) # TODO swap to np.angle instead of arctan2
	
	dt=(da.t[1]-da.t[0]).data
	daFreq=_xr.DataArray(	_np.gradient(_np.unwrap(daPhase),dt)/(2*_np.pi),
							   dims=['t'],
							   coords={'t':da.t})
	if plot==True:
		
		fig,ax=_plt.subplots(3,sharex=True)
		_np.real(daHilbert).plot(ax=ax[0],label='original')
		_np.imag(daHilbert).plot(ax=ax[0],label=r'90$^o$ shifted')
		daAmp.plot(ax=ax[0],label=r'amplitude')
		daPhase.plot(ax=ax[1],label=r'phase',linestyle='',marker='.')
		daFreq.plot(ax=ax[2],label=r'freq')

		_finalizeSubplot(ax[0],
								ylabel='Amplitude',
								subtitle='Signals',
								legendLoc='lower right')
		_plot.finalizeSubplot(ax[1],
								ylabel='Rad.',
								subtitle='Phase',
								legendOn=False,
								yticks=[-_np.pi,-_np.pi/2,0,_np.pi/2,_np.pi],
								ytickLabels=[r'-$\pi$',r'-$\pi$/2',r'0',r'$\pi$/2',r'$\pi$'],
								ylim=[-_np.pi,_np.pi],
								xlabel='Time',)
		_plot.finalizeSubplot(ax[2],
								ylabel='Hz',
								subtitle='Frequency',
								xlabel='Time',
								legendOn=False)
		
	return daHilbert,daAmp,daPhase,daFreq




