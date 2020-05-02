
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from scipy import fftpack as _fftpack
from johnspythonlibrary2 import Plot as _plot

def stft(	df,
			numberSamplesPerSegment=1000,
			numberSamplesToOverlap=500,
			frequencyResolutionScalingFactor=1.,
			plot=False,
			verbose=True,
			logScale=False):
	"""
	Short time fourier transform across a range of frequencies
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		1D DataFrame of signal
		index = time
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
	dfResult : pandas dataframe
		index is time. columns is frequency.  values are the complex results at each time and frequency.	
	
	Examples
	--------
	Example1::
		
		# create fake signal.
		import numpy as np
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
		df = _pd.DataFrame(x,index=time)
		
		# function call
		dfResult=stft(df,plot=True)
		
		
	Example2::
		
		# create fake signal.
		import numpy as np
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
		df = _pd.DataFrame(x,index=time)
		
		# function call
		dfResult=stft(df,plot=True)
		
	Notes
	-----
		1. The sampling rate sets the upper limit on frequency resolution
		2. numberSamplesPerSegment sets the lower limit on the (effective) frequency resolution 
		3. numberSamplesPerSegment sets an (effective) upper limit on the time responsiveness of the algorithm (meaning, if the frequency rapidly shifts from one value to another)

	"""
	from scipy.signal import stft as scipystft
	import numpy as np
	
	if type(df)==_pd.core.series.Series:
		df=_pd.DataFrame(df)
	
	dt=df.index[1]-df.index[0]
	fs=1./dt
	
	if verbose:
		
		print("Sampling rate: %.3e Hz"%fs)
		
		timeWindow=dt*numberSamplesPerSegment
		print("Width of sliding time window: %.3e s"%timeWindow)
	
		# lowest frequency to get at least one full wavelength
		fLow=1./(numberSamplesPerSegment*dt)
		print("Lowest freq. to get at least one full wavelength: %.2f" % fLow )
	
		# frequency upper limit
		nyF=fs/2.
		print("Nyquist freq. (freq. upperlimit): %.2f" % nyF)
	
	fOut,tOut,zOut=scipystft(df.iloc[:,0].values,fs,nperseg=numberSamplesPerSegment,noverlap=numberSamplesToOverlap,nfft=df.shape[0]*frequencyResolutionScalingFactor)
	zOut*=2 # TODO(John) double check this scaling factor.   Then cite a reason for it.  (I don't like arbitrary factors sitting around)
	tOut+=df.index[0]
	
	dfResult=_pd.DataFrame(zOut.transpose(),index=tOut,columns=fOut)

	if plot==True:
		
		if logScale==False:
			fig,ax,cax=_plot.subplotsWithColormaps(1)
			levels=np.linspace(0,dfResult.abs().max().max(),61)
			pc=ax.contourf(dfResult.index,dfResult.columns,np.abs(dfResult.values.transpose()),levels=levels,cmap='Blues')
			fig.colorbar(pc,ax=ax,cax=cax)
			_plot.finalizeSubplot(ax,
								 xlabel='Time (s)',
								 ylabel='Frequency (Hz)',
								 legendOn=False)
			_plot.finalizeFigure(fig)
		else:
			raise Exception('Not implemented yet...')
			#TODO implement stft spectrogram plot with logscale on the zaxis (colorbar) 
			# starting reference : https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/contourf_log.html
			
			
	return dfResult
	

def stftSingleFrequency_df(df,
						   freq,
						   windowSizeInWavelengths=2,
						   plot=False,
						   verbose=True,):
	"""
	Short-time Fourier transform of a single frequency.  Uses a Hann window
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		columns contain signals
		index = time
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
		
		dt=2e-6
		t=np.arange(0,10e-3,dt)
		f=1.5e3
		y=np.sin(2*np.pi*t*f+0.25*np.pi)
		
		N=t.shape[0]
		n=np.arange(0,N)
		hannWindow=np.sin(np.pi*n/(N-1.))**2
		hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
		
		df=pd.DataFrame(np.array([y,y*1.1,y*1.2]).transpose(),index=t,columns=['a','b','c'])
		
		dfComplex,dfAmp,dfPhase=stftSingleFrequency_df(df,f,plot=True)
	"""
	
	import numpy as np
	
	# initial calculations
	dt=df.index[1]-df.index[0]
	fs=1./dt
	N=np.ceil(windowSizeInWavelengths*fs/freq).astype(int)
	dN=1
	M=df.shape[0]
	
	# calculate hann window
	n=np.arange(0,N)
	hannWindow=np.sin(np.pi*n/(N-1.))**2
	hannWindow/=np.sum(hannWindow)*1.0/N  	# normalize
	dfHann=_pd.DataFrame([hannWindow]*df.shape[1]).transpose()
			
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
	dfAmp=_pd.DataFrame(index=df.index[steps+int(N/2)],
							columns=df.columns,
							dtype=complex)
	dfComplex=_pd.DataFrame(index=df.index[steps+int(N/2)],
							columns=df.columns,
							dtype=complex)
	dfPhase=_pd.DataFrame(index=df.index[steps+int(N/2)],
							columns=df.columns,
							dtype=float)
	
	# perform analysis
	#TODO optomize this section of code.  should be a convolution function somewhere
	for i in range(0,len(steps)):
		temp=_pd.DataFrame(df.iloc[steps[i]:steps[i]+N].values,
					index=dt*(n-n.mean()),
					columns=df.columns)
		dfOut=fftSingleFreq_df(temp*dfHann.values,freq,plot=False)
			
		dfComplex.iloc[i,:]=dfOut.loc['fft']
		dfAmp.iloc[i,:]=dfOut.loc['amp']
		dfPhase.iloc[i,:]=dfOut.loc['phase']
		
	if plot==True:
		for i,(key,val) in enumerate(df.iteritems()):
			
			fig,(ax1,ax2)=_plt.subplots(2,sharex=True)
			ax1.plot(val)
			ax1.plot(dfAmp[key])
			ax2.plot(dfPhase[key],'.')
		
	return dfComplex,dfAmp,dfPhase



	
	
def _coherenceComplex(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
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


def coherenceAnalysis(t,y1,y2,numPointsPerSegment=1024,plot=False,noverlap=None,verbose=True):
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
		
		import numpy as np
		freq=5
		time = np.arange(0, 10, 0.01)
		sinewave1 = np.sin(time*np.pi*2*freq)+ 3.*np.random.randn(len(time)) # add white noise to the signal
		sinewave2 = np.sin(time*np.pi*2*freq+1.)
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
		ax1.plot(t,y1,label='Raw 1')
		ax1.plot(t,y2,label='Raw 2')
		ax1.legend()
		ax1.set_xlabel('time (s)')
		ax2.plot(f, np.abs(Cxy),'.-',label='Coherence')
		# ax2.semilogy(f, Cxy,label='Coherence')
		ax2.set_xlabel('frequency (Hz)')
		ax2.set_ylabel('Coherence')
		ax2.legend()
		ax2.set_ylim([0,1])
		
	return f, Cxy


def ifft(	dfFFT,
			t=None,
			plot=False,
			invertNormalizedAmplitude=True,
			returnRealOnly=True):
	
	"""
	Examples
	--------
	
	Example1::
		
		import numpy as np
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
							index=t,
							columns=['a','b','c'])
		dfFFT=fft_df(df,plot=False,normalizeAmplitude=False)
		
		df2=ifft(dfFFT)
		
		for key,val in df.iteritems():
			
			_plt.figure()
			_plt.plot(df.index,val)
			sig=df2[key]
			sig2=_pd.DataFrame(sig.abs())*np.exp(phase(sig))
			_plt.plot(df2.index,sig)
			
			
	Example2::
			
		import pandas as pd
		import numpy as np
		import johnspythonlibrary2 as jpl2
		
		# input signal
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y1=jpl2.Process.SigGen.chirp(t,[1e-3,9e-3],[1e3,1e5])
		df1=pd.DataFrame(y1,index=t)
		
		# filter type 1 : butterworth filter
		y2=jpl2.Process.Filters.butterworthFilter(df1,10e3,plot=False)
		df2=pd.DataFrame(y2,index=t)
		
		# filter type 2 : IFFT reconstructed Butterworth filter
		tf=jpl2.Process.TF.calcTFFromSingleTimeSeriesSpanningMultipleFrequencies(df1,df2,plot=False)
		dfFFT=fft_df(df1,plot=False,normalizeAmplitude=False)
		df3=ifft(pd.DataFrame(tf['lowpassFiltered']*dfFFT[0]))
		
		# plots
		fig,ax=plt.subplots(2,sharex=True)
		ax[0].plot(df1,label='Input signal')
		ax[0].plot(df2,label='Output signal')
		ax[1].plot(df1,label='Input signal')
		ax[1].plot(df3,label='Output signal')
		_plot.finalizeSubplot( 	ax[0],
								subtitle='Butterworth filter')
		_plot.finalizeSubplot( 	ax[1],
								subtitle='IFFT reconstructed Butterworth filter')
		
	"""
	
	if type(dfFFT)==_pd.core.series.Series:
		dfFFT=_pd.DataFrame(dfFFT)
	
	# create a time basis if not provided
	if t==None:
		N=dfFFT.shape[0]
		df=dfFFT.index[1]-dfFFT.index[0]
		dt=1/df/N
		t=_np.arange(0,N)*dt
		
	# IFFT function
	dfIFFT=dfFFT.apply(_fftpack.ifft,axis=0).set_index(t)
	
	# option
	if returnRealOnly==True:
		dfIFFT=_pd.DataFrame(_np.real(dfIFFT),index=dfIFFT.index,columns=dfIFFT.columns)
	
	# plots
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(dfIFFT)
		
	return dfIFFT
	
	
	
def phase(df):
	"""
	Calculates the phase of complex data series.
	Basis can be time, frequency, etc.
	"""
	
	if type(df)==_pd.core.series.Series:
		df=_pd.DataFrame(df)
	
	return _pd.DataFrame(_np.arctan2(_np.imag(df),_np.real(df)),
						      index=df.index,
							  columns=df.columns)
	

	
def fft_df(df,plot=False,trimNegFreqs=False,normalizeAmplitude=False):
	"""
	Simple wrapper for fft from scipy
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		dataframe of time dependent data
		index = time
	plot : bool
		(optional) Plot results
	trimNegFreqs : bool
		(optional) True - only returns positive frequencies
	normalizeAmplitude : bool
		(optional) True - normalizes the fft (output) amplitudes to match the time series (input) amplitudes
		
	Returns
	-------
	dfFFT : pandas.core.frame.DataFrame
		complex FFT of df
	
	References
	-----------
	https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
	
	Example
	-------
	::
		
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
							index=t,
							columns=['a','b','c'])
		dfFFT=fft_df(df,plot=True,trimNegFreqs=False,normalizeAmplitude=False)
	"""
	
	if type(df)!=_pd.core.frame.DataFrame:
		if type(df)==_pd.core.series.Series:
			df=_pd.DataFrame(df)
		else:
			raise Exception('Input data not formatted correctly')
	
	# initialize
	
	# fft
	dt=df.index[1]-df.index[0]
	freq = _fftpack.fftfreq(df.shape[0],d=dt)
	dfFFT=df.apply(_fftpack.fft,axis=0).set_index(freq)
	
	# options
	if trimNegFreqs==True:
		dfFFT=dfFFT[dfFFT.index>=0]
	if normalizeAmplitude==True:
		N=df.shape[0]
		dfFFT*=2.0/N
		
	# optional plot of results
	if plot==True:
		
		dfAmp=dfFFT.abs()
		dfPhase=phase(dfFFT)
		for i,(key,val) in enumerate(df.iteritems()):
			f,(ax1,ax2,ax3)=_plt.subplots(nrows=3)
			
			ax1.plot(val)
			ax1.set_ylabel('Orig. signal')
			ax1.set_xlabel('Time')
			ax1.set_title(key)
			
			ax2.plot(dfPhase.index,dfAmp[key],marker='.')
			ax2.set_ylabel('Amplitude')
			
			ax3.plot(dfPhase.index,dfPhase[key],marker='.',linestyle='')
			ax3.set_ylabel('Phase')
			ax3.set_xlabel('Frequency')
			ax3.set_ylim([-_np.pi,_np.pi])
		
	# return results
	return dfFFT



def fftSingleFreq_df(df,f,plot=False):
	"""
	Performs a Fourier transform of a signal at a single frequency
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		the signal being analyzed
		index = time
	f : float
		The frequency (in Hz) that is being investigated
		
	Return
	------
	dfResults : pandas.core.frame.DataFrame
		fft results at frequency, f
		
	References
	----------
	https://dsp.stackexchange.com/questions/8611/fft-for-a-single-frequency
	
	Example
	-------
	::
		
		dt=2e-6
		f=1e3
		t=np.arange(0,10e-3,dt)
		y=np.sin(2*np.pi*t*f)
		df=_pd.DataFrame( 	np.array([y,y*1.1,y*1.2]).transpose(),
							index=t,
							columns=['a','b','c'])
		dfResults=fftSingleFreq_df(df,f,plot=True)

	"""
	
	# init
	N=df.shape[0]
	
	# fft (complex, amplitude, and phase)
	dfResults=_pd.DataFrame(columns=df.columns,index=['fft','amp','phase'])
	for i,(key,val) in enumerate(df.iteritems()):
#		print(key)
		fft=_np.nansum(val*_np.exp(-1j*2*_np.pi*f*val.index.values))*2./N
		dfResults.at['fft',key]=fft
		amp=_np.abs(fft)
		dfResults.at['amp',key]=amp
		phase=wrapPhase(_np.arctan2(_np.imag(fft),_np.real(fft))+_np.pi/2)
		dfResults.at['phase',key]=phase
#		print(phase)

		# optional plot
		if plot==True:
			fig,ax=_plt.subplots()
			ax.set_title("%s\nFreq: %.1f, Avg. amplitude: %.3f, Avg. phase: %.3f"% (key,f,amp,phase))
			ax.plot(val)
			ax.plot(val.index,amp*_np.ones(val.shape[0])) 
		
	return dfResults



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
	dfPhase=_pd.DataFrame(_np.arctan2(_np.imag(dfHilbert),_np.real(dfHilbert)),
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



def wrapPhase(phases):
	""" 
	simple wrap phase function from -pi to +pi
	
	Parameters
	----------
	phases : numpy.ndarray or pandas.core.frame.DataFrame
		array of phase data
		
	Returns
	-------
	wrapped phase
	
	Example
	-------
	phi=np.arange(-10,10,0.1)
	fig,ax=plt.subplots()
	ax.plot(phi,wrapPhase(phi),'x')
	plt.show()
	
	References
	----------
	https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
	"""
	return (phases + _np.pi) % (2 * _np.pi) - _np.pi
		
	
	
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



def bicoherence(	sx,
					windowLength,
					numberWindows,
					plot=False,
					windowFunc='Hann',
					title=''):
	"""
	Bicoherence and bispectrum analysis.  This algorithm is based on [Kim1979].
	
	Parameters
	----------
	sx : pandas.core.series.Series
		Signal.  index is time.
	windowLength : int
		Length of each data window
	numberWindows : int
		Number of data windows
	plot : bool
		Optional plot of data
	windowFunc : str
		'Hann' uses a Hann window (Default)
		Otherise, uses no window 
		
	Returns
	-------
	dfBicoh : pandas.core.frame.DataFrame
		Bicoherence results.  Index and columns are frequencies.
	dfBispec : pandas.core.frame.DataFrame
		Bispectrum results.  Index and columns are frequencies.
	
	References
	----------
	* Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979). 

	* D.Kong et al Nuclear Fusion 53, 113008 (2013).


	"""
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	
	N=windowLength
	M=numberWindows
		
	### subfunctions	
	def FFT(s,plot=False,shift=False):
		""" Wrapper for the scipy fft algorithm """
		N=s.shape[0]
		dt=s.index[1]-s.index[0]
		from scipy.fft import fft, fftfreq, fftshift
		xi=s.values.reshape(-1)
		xi-=xi.mean()
		ti=s.index.values-s.index.values[0]
		
		if shift==True:
			X=fftshift(fft(xi)*2/N)
			f = fftshift(fftfreq(ti.shape[-1]))/dt
			sX=pd.Series(X,index=f)#.iloc[1:,:]
		else:
			X=fft(xi)[0:N//2]*2/N
			f = fftfreq(ti.shape[-1])[0:N//2]/dt
			sX=pd.Series(X,index=f)
			
		if plot:
			fig,ax=plt.subplots()
			ax.semilogy(sX.abs(),marker='.')
		
		return sX
	
	
	def plot_2D(dfB,fig,ax,cax,title='Bispectrum'):
		""" Plot the bispectrum """
		
		fx=dfB.columns.values
		fy=dfB.index.values
		Fx,Fy=np.meshgrid(fx,fy)
		draw=ax.pcolormesh(Fx,Fy,dfB.abs().values,vmin=0,vmax=1)
		plt.colorbar(draw,cax=cax,ax=ax)
		ax.axis('equal')
		ax.set_xlabel(r'$f_1$ (Hz)')
		ax.set_ylabel(r'$f_2$ (Hz)')
		ax.set_title(title)


	### main code
	
	# calculate window function
	if windowFunc=='Hann':
		n=np.arange(0,N)
		window=np.sin(np.pi*n/(N-1.))**2
		window/=np.sum(window)*1.0/N  	# normalize
	else:
		window=np.ones(N)
		window/=np.sum(window)*1.0/N  	# normalize   ???
		
	# step in time
	for i in range(M):
		
		# window data
		index=np.arange(N*(i),N*(i+1),)
		sxi=(sx.iloc[index]-sx.iloc[index].mean())*window
		
		# fft 
		sXi=FFT(sxi,shift=True,plot=False)
		
		# calculate Xk and Xl
		sXk=sXi[sXi.index>=0]
		sXl=sXi.copy()
		Xk,Xl=np.meshgrid(sXk,sXl)
		
		# misc parameters for later
		f=sXi.index.values
		df=f[1]-f[0]
		q=f[-1]
		K=f[f>=0]
		L=f
		Kmesh,Lmesh=np.meshgrid(K,L)
		A,B=np.meshgrid(	np.arange(0,len(K)),
						    np.arange(0,len(L)))
		C=A+B
		
		# calculate Xkl
		dfXTemp=sXi.append(pd.DataFrame(np.zeros(N)*np.nan,index=f-f[0]+f[-1]+df))
		Xkl=dfXTemp.iloc[C.reshape(-1),0].values.reshape(N,N//2)

		# initialize dataframes and mask on first iteration
		if i==0:
			
			# dataframes
			dfBispec=pd.DataFrame(np.zeros((len(L),len(K))),index=L,columns=K,dtype=complex)
			dfDenom1=pd.DataFrame(np.zeros((len(L),len(K))),index=L,columns=K,dtype=complex)
			dfDenom2=pd.DataFrame(np.zeros((len(L),len(K))),index=L,columns=K,dtype=complex)
			
			# create mask
			inRegionA= ((0<=Lmesh) & (Lmesh<=q/2) & (Lmesh<=Kmesh) & (Kmesh<=q-Lmesh))
			inRegionB= ((-q<=Lmesh) & (Lmesh<=0) & (Kmesh>np.abs(Lmesh)) & (Kmesh<=q))
			mask=(inRegionA | inRegionB).astype(float)
			mask[mask==False]=np.nan
	
		# main calculations for each time step.
		dfBispec+=Xl*Xk*np.conjugate(Xkl)#*np.conjugate(Xkl).values
		dfDenom1+=np.abs(Xl*Xk)**2
		dfDenom2+=np.abs(Xkl)**2
		
	# apply mask and trim frequency domain
	dfBicoh=dfBispec**2*mask/(dfDenom1*dfDenom2)
	dfBicoh=dfBicoh[dfBicoh.index<=(q+df)/2]
	dfBispec=dfBispec*mask/M
	dfBispec=dfBispec[dfBispec.index<=(q+df)/2]
					
	# optional plots
	if plot==True:
		
		fig, ax = plt.subplots(1,2,sharex=False)
		cax=[]
		for axi in ax:
			divider=make_axes_locatable(axi)
			cax.append(divider.append_axes("right", size="2%", pad=.05))
		
		plot_2D(dfBicoh,fig,ax[0],cax[0],title='Bicoherence')
		
		cax[1].remove()
		ax[1].plot(FFT(sx.iloc[0:N]).abs())
		ax[1].set_title('Power spectrum')
		ax[1].set_xlabel('Hz')
		
		fig.suptitle(title)
		fig.tight_layout(h_pad=0.25,w_pad=3,pad=0.5) # sets tight_layout and sets the padding between subplots
			
			
		if plot=='all':
			plot_2D(dfBispec,title='Bispectrum')
			plt.figure();plt.plot(sx);
		
		
	return dfBicoh,dfBispec




def calcPhaseDifference(dfX1,dfX2,plot=False,title=''):
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
		p1=_np.arctan2(_np.imag(dfX1),_np.real(dfX1))
		p2=_np.arctan2(_np.imag(dfX2),_np.real(dfX2))
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




