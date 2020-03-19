
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt



def butterworthFilter(y, x,filterOrder=2, samplingRate=1/(2*1e-6), 
					  cutoffFreq=20*1e3, filterType='low',plot=False):
	"""
	Apply a digital butterworth filter on your data
	
	Parameters
	----------
	y : numpy.ndarray
		unfiltered dependent data
	x : numpy.ndarray
		independent data
	filterOrder : int
		Butterworth filter order
	samplingRate : float
		Data sampling rate.  
	cutoffFreq : float
		cutoff frequency for the filter
	filterType : str
		filter type.  'low' is lowpass filter
	plot : bool or str
		- True - plots filter results. 
		- 'all'- plots filter results and filter response (psuedo-BODE plot)
		
	Returns
	-------
	filteredData : numpy.ndarray
		Filtered dependent data
		
	References
	----------
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
	https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
	"""	
	
	
	from scipy.signal import butter, lfilter#, freqz
	
	def butter_lowpass(cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = butter(order, normal_cutoff, btype=filterType, analog=False)
		return b, a
	
	def butter_lowpass_filter(data, cutoff, fs, order=5):
		b, a = butter_lowpass(cutoff, fs, order=order)
		y = lfilter(b, a, data)
		return y
		
		
	filteredData=butter_lowpass_filter(y, cutoffFreq,
									   samplingRate, filterOrder)
									   
		
	return filteredData
		
