
import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr

from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot, finalizeFigure as _finalizeFigure


def fftSignalReconstruct(ds,numModes=None,plot=False):
	"""
	Creates a transfer function between an input and output signal (i.e. tf=H(s)=FFT(s1B)/FFT(s1A)) and applies it to s2A to reconstruct s2B.  

	Parameters
	----------
	ds : xr.Dataset
		s1A : xr.DataArray
			Input signal used to create the transfer fuction.  Time series.
		s2A : xr.DataArray
			Input signal that is used to reconstruct s2B with tf
		s1B : xr.DataArray
			Output signal used to create the transfer fuction
		s2B : xr.DataArray
			(Optional) The actual s2B signal.  If provied, it will be plotted alongside the reconstructed s2B for comparison. The anaylsis does not use it otherwise. 
	numModes : int
		The first N number of frequencies to use with the reconstruction.
		If None, the code uses all frequencies.
	plot : bool
		True - Provides an optional plot of the results

	Returns
	-------
	ds : xr.Dataset
		Same as input with the addition of
		s2B_recon : pandas.core.series.Series
			The reconstructed s2B signal.  Time series.
		
	Examples
	--------
	::
		
		from johnspythonlibrary2.Process.SigGen import chirp
		from johnspythonlibrary2.Process.Filters import butterworth
		from johnspythonlibrary2.Process.SigGen import gaussianNoise
		import xarray as _xr
		import numpy as _np
		
		t=_np.arange(0,20e-3,2e-6)
		fStart=1e3
		fStop=1e5
		s2A=_xr.DataArray(	chirp(t,[0.5e-3,19.46e-3],[fStart,fStop]),
							dims=['t'],
							coords={'t':t})
		s1A=_xr.DataArray(	gaussianNoise(t.shape),
							dims=['t'],
							coords={'t':t})
		f_corner=1e4
		s1B=_xr.DataArray(	butterworth(s1A,f_corner,filterType='low',plot=False),
							dims=['t'],
							coords={'t':t})
		ds=_xr.Dataset({'s1A':s1A,
				  's1B':s1B,
				  's2A':s2A,})
		s2B_recon=fftSignalReconstruct(ds,plot=True)
		
		
		from johnspythonlibrary2.Process.Spectral import fft
		bodePlotFromTF(fft(s2A))
		bodePlotFromTF(fft(s2B_recon))
		

	"""
	if 's1A' not in ds._variables.keys() or 's1B' not in ds._variables.keys() or 's2A' not in ds._variables.keys():
		raise Exception('Input data not formatted correctly')
	
	# calculate TF from the first half of signals 1 and 2 (i.e. s1A and s2A)
	tf=calcTF(ds.s1A,ds.s1B,plot=False)

	# sanity check - reconstruct s1B from the TF
	if plot==True:
		_fftSignalReconstructFromTF(	ds.s1A,
										tf,
										plot=True,
										da2=ds.s1B)
		
	def trimFreqs(da,numF=50):
		fmax=da.f[50].data
		da[(da.f>fmax) | (da.f<-fmax)]=0
		return da
 
	# trim tf based on the number of modes (unique frequencies) to maintain.
	if type(numModes)!=type(None):
		tf_trimmed=trimFreqs(tf.copy(),numModes)
	else:
		tf_trimmed=tf.copy()
		
	if 's2B' in ds._variables.keys():
		s2B = ds.s2B
	else:
		s2B = None
	s2B_recon=_fftSignalReconstructFromTF(	ds.s2A,
											tf_trimmed,
											plot=plot,
											da2=s2B)
		
	return s2B_recon


def _fftSignalReconstructFromTF(da1,tf,da2=None,plot=False,positiveFreqsOnly=True):
	""" 
	Reconstructs a time-base signal (s2_recon) from an input signal (s1) and a transfer function (tf) 
	
	Parameters
	----------
	da1 : xr.DataArray
		Input signal.  
	tf : xr.DataArray
		Transfer function.  
	da2 : xr.DataArray
		(Optional) Original output signal.  Include it if you want it plotted alongside the reconstructed s2 for comparison
	plot : bool
		(Optional) Plot of results
		
	Returns
	-------
	da2_recon : xr.DataArray
		Output signal.  
	"""
	from johnspythonlibrary2.Process.Spectral import fft,ifft
	
	# take fft of s1
	X=fft(da1)
	
	# reconstruction:  take ifft of tf multiplied by X.  
	da2_recon=ifft(tf*X,t=da1.t.data)
	
	if plot==True:
		
		fig,ax=_plt.subplots(2,sharex=True)
		da1.plot(ax=ax[0],label='signal 1')
		da2_recon.plot(ax=ax[1],label='signal 2 recon')
		if type(da2)!=type(None):
			da2.plot(ax=ax[1],label='signal 2')
		_finalizeSubplot(ax[0],title='signal reconstruction')
		_finalizeSubplot(ax[1])
		_finalizeFigure(fig)
		
	return da2_recon


def _dB(y,yRef=1.0,dbScale=20.0):
	""" calculate dB (decibel) of any number/array """
	return dbScale*_np.log10(y/yRef)


def _dBInverse(dBSignal,dbScale=20.0):
	""" inverse calculation of dB to standard gain ratio """
	return 10.0**(dBSignal/dbScale)


# def phaseCalcFromComplexSignal(daComplex,plot=False):
# 	"""
# 	Calculates phase from a complex signal using the arctan2() function.
# 	
# 	Parameters
# 	----------
# 	dfComplex : xarray.DataArray
# 		Complex signal. 
# 	plot : bool
# 		Optional plot
# 		
# 	Returns
# 	-------
# 	dfPhase : xarray.DataArray
# 		Dataframe containing the phase results.
# 		Index will be the same as dfComplex
# 		
# 	Examples
# 	--------
# 	Example1::
# 		
# 		from johnspythonlibrary2.Process.SigGen import chirp

# 		t=_np.arange(0,20e-3,2e-6)
# 		fStart=2e2
# 		fStop=2.0e4
# 		y1=_chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
# 		da=_xr.DataArray(y1,
# 					  dims=['t'],
# 					  coords={'t':t}) 
# 		
# 		from johnspythonlibrary2.Process.Spectral import fft
# 		
# 		X=fft(da)
# 		
# 		phaseCalcFromComplexSignal(X,plot=True)
# 		
# 	"""
# 	
# 	daPhase=	_np.arctan2(_np.imag(daComplex),_np.real(daComplex))
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots(2,sharex=True)
# 		_np.real(daComplex).plot(ax=ax[0],label='real')
# 		_np.imag(daComplex).plot(ax=ax[0],label='imag')
# 		daPhase.sortby('f').plot(ax=ax[1],label='phase')
# 		
# 		_finalizeSubplot(	ax[0],
# 							  subtitle='Input',
# 							  )
# 		_finalizeSubplot(	ax[1],
# 							  subtitle='Phase',
# 							  )
# 		
# 	return daPhase
	

def bodePlotFromTF(	daTF,
					fig=None,
					ax=None,
					label=None,
					degScaleForPhase=True,
					dBScaleForAmplitude=False,
					semilogXAxis=False,
                    alpha=None):
	"""
	Creates a bode plot from a provide transfer function.
	
	Parameters
	----------
	dfTP : pandas.core.frame.DataFrame
		Transfer function in the fourier/laplace space.  Index is frequency with units of Hz.
	fig : matplotlib.figure.Figure, optional
		Figure to create the plot.
	ax : list containing two matplotlib.axes._subplots.AxesSubplot
		The two axes to create the plot
	degScaleForPhase : bool
		True - phase plot has units in degrees
		False - phase plot has units in radians
	dBScaleForAmplitude : bool
		True - magnitude plot has is converted to dB
		False - amplitude plot has is an uncorrected ratio
	semilogXAxis : bool
		True - log scale on x-axis
		False - linear scale on x-axis
		
	Returns
	-------
	fig : matplotlib.figure.Figure, optional
		Figure to create the plot.
	ax : list containing two matplotlib.axes._subplots.AxesSubplot
		The two axes to create the plot
	
	"""
	
	if 'float' not in str(daTF.f.dtype):
		raise Exception('daTF frequency should be data type float.  Instead \'%s\' found'%(daTF.f.dtype))
		
	daTF=daTF.copy()
	daTF=daTF.sortby('f')
		
	if fig==None:
		fig,ax=_plt.subplots(2,sharex=True)
	
	# amplitude
	amp=_np.abs(daTF)
	if dBScaleForAmplitude==False:
		y0Label='unitless (ratio)'
		pass
	else:
		amp=_dB(amp)
		y0Label=r'unitless (dB)'
	#amp.plot(ax=ax[0],linestyle='-', label=label)
	amp.plot(ax=ax[0],linestyle='', marker='.', label=label, alpha=alpha)
	
	# phase
	phase = _np.angle(daTF)
	phase = _xr.DataArray(phase, coords=daTF.coords)
	if degScaleForPhase==True:
		phase*=180/_np.pi
		y1Label='deg.'
		y1Lim=[-180,180]
		y1Ticks=_np.array([-180,-90,0,90,180])
		y1TicksLabels=[]
		#y1TicksLabels=_np.array([-180,-90,0,90,180])
		#y1TicksLabels=_np.array([r'$-180^o$',r'$-90^o$',r'$0^o$',r'$90^o$',r'$180^o$'])
	else:
		y1Label='rad.'
		y1Lim=[-_np.pi,_np.pi]
		y1Ticks=[-_np.pi,-_np.pi*0.5,0,_np.pi*0.5,_np.pi]
		y1TicksLabels=[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$']
		pass
	phase.plot(ax=ax[1],linestyle='',marker='.', label=label, alpha=alpha)
	
	# finalize plot
# 	_finalizeSubplot(	ax[0],
# 						ylabel=y0Label,
# 						subtitle='Magnitude',
# 						yscale='log'
# 						)
# 	_finalizeSubplot(	ax[1],
# 						ylabel=y1Label,
# 						xlabel='Frequency (Hz)',
# 						subtitle='Phase diff.',
# 						ylim=y1Lim,
# 						yticks=y1Ticks,
# 						ytickLabels=y1TicksLabels,
# 						)
	ax[0].set_yscale('log')
	if semilogXAxis==True:
		ax[0].set_xscale('log')
		ax[0].set_xlim([0,amp.f.max()])
	_finalizeFigure(	fig)
	
	return fig,ax
	
	


def calcTF(	daInput,
			daOutput,
			plot=False):
	"""
	Calculated the transfer function from one or more input signals
	If daInput and daOutput have 'fRange' as the second coordinate, the calcTF function will calcTF for each frequency in the fRange dimension.
	
	Parameters
	----------
	daInput : xarray.DataArray
		The original (reference) signal that goes into your system.
		dim[0] = 't' or time
		(optional) dim[1] = 'fRange' for each frequency to perform the calculation
	daOutput : xarray.DataArray
		The modified  signal that comes out of your system.
		dim[0] = 't' or time
		(optional) dim[1] = 'fRange' for each frequency to perform the calculation
	plot : bool
		Optional plot of results
		
	Returns
	-------
	daTF : xarray.DataArray
		Transfer function.  
		dim[0] is frequency
	
	Examples
	--------
	Example1::
		
		### Single frequency sweep 
		
		from johnspythonlibrary2.Process.SigGen import chirp
		from johnspythonlibrary2.Process.Filters import butterworth
		import xarray as _xr
		import numpy as _np
		
		t=_np.arange(0,20e-3,2e-6)
		fStart=1e2
		fStop=1e6
		y1=chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
		da=_xr.DataArray(y1,
					  dims=['t'],
					  coords={'t':t}) 
		fwhm=0.5e-3
		f_corner=1e4
		daHP=butterworth(da,f_corner,filterType='high',plot=False)
		daLP=butterworth(da,f_corner,filterType='low',plot=False)
		
		if False:
			fig,ax=_plt.subplots()
			ax.plot(da)
			ax.plot(daHP,label='HP')
			ax.plot(daLP,label='LP')
			ax.legend()
	
		tf1=calcTF(daInput=da,daOutput=daHP,plot=True)
		tf2=calcTF(daInput=da,daOutput=daLP,plot=True)
		
		tf1=tf1.sortby('f')
		tf2=tf2.sortby('f')
		
		tf1=tf1[(tf1.f>=fStart)&(tf1.f<=fStop)]
		tf2=tf2[(tf2.f>=fStart)&(tf2.f<=fStop)]
		fig,ax=_plt.subplots(2,sharex=True)
		bodePlotFromTF(tf1,fig,ax,semilogXAxis=True, label='Highpass')
		bodePlotFromTF(tf2,fig,ax,semilogXAxis=True, label='Lowpass')
		ax[0].set_xlim([tf1.f[0],tf1.f.max()])

		
	Example2::
		
		### Frequency sweep at several static frequencies
		
		from johnspythonlibrary2.Process.SigGen import chirp
		from johnspythonlibrary2.Process.Filters import butterworth
		import xarray as _xr
		import numpy as _np
		
		t=_np.arange(0,20e-3,2e-6)
		fStart=1e2
		fStop=1e6
		y1=chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
		fRange=10**_np.arange(2,6+0.031,0.031)
		daInput=_xr.DataArray(
					  dims=['t','fRange'],
					  coords={'t':t,
							   'fRange':fRange}) 
		daOutput=_xr.DataArray(
					  dims=['t','fRange'],
					  coords={'t':t,
							   'fRange':fRange}) 
		f_corner=1e4
		for f in fRange:
			daInput.loc[:,f]=y1
			daOutput.loc[:,f]=butterworth(daInput.sel(fRange=f),f_corner,filterType='high',plot=False)

		calcTF(daInput,daOutput,plot=True)
		_plt.gca().set_xlim([fStart,fStop])
		_plt.gca().set_xscale('log')
		
	Example3::
		
		print('work in progress')
		from johnspythonlibrary2.Process.SigGen import chirp
		import johnspythonlibrary2 as jpl2
		import numpy as _np
		import matplotlib.pyplot as _plt
		
		t =_np.arange(0,1e0,1e-7)
		f_lim = [1e2, 1e6]
		y_original = chirp(t, [t[0], t[-1]], fStartStop=f_lim, plot=False)
		f_cutoff = 1.00123e4
		
		plot = False
		y_boxcar = jpl2.Process.Filters.boxcar_convolution_filter(y_original, width_in_time = 1/f_cutoff, plot=plot)
		y_gaussian = jpl2.Process.Filters.gaussianFilter(y_original, 1/f_cutoff, 'low', plot=plot )

		tf_boxcar = calcTF(y_original, y_boxcar)
		tf_gaussian = calcTF(y_original, y_gaussian)
		
		fig, ax = _plt.subplots(2, sharex=True)
		bodePlotFromTF(tf_boxcar, fig=fig, ax=ax, label='boxcar, lp', semilogXAxis=True)
		bodePlotFromTF(tf_gaussian, fig=fig, ax=ax, label='gaussian, lp', semilogXAxis=True)
		ax[0].set_xlim(f_lim)
		ax[0].axvline(f_cutoff, color='r', linestyle='--', label='f_cutoff')
		ax[0].legend()
		
	"""
	
	from johnspythonlibrary2.Process.Spectral import fft
	
	# standard calculation
	if 'fRange' not in daInput.dims:
		Xin=fft(daInput,trimNegFreqs=True)
		Xout=fft(daOutput,trimNegFreqs=True)
		
		daTF=_xr.DataArray(	Xout/Xin,
							 dims='f',
							 coords={'f':Xin.f})
		
		if plot==True:
			fig,ax=_plt.subplots(2,sharex=True)
			
			bodePlotFromTF(	daTF.sortby('f'),
							fig,
							ax,
							dBScaleForAmplitude=False)
		
		return daTF
	
	
	# fixed freq step calculations
	else:
		
		from johnspythonlibrary2.Process.Spectral import fftSingleFreq
		
		daTF=_xr.DataArray(	_np.zeros(daInput.fRange.shape,dtype=complex),
							 dims=['f'],
							 coords={'f':daInput.fRange.data},
							 )
		
		for f in daInput.fRange.data:
 			Xin=fftSingleFreq(daInput.sel(fRange=f),f)
 			Xout=fftSingleFreq(daOutput.sel(fRange=f),f)
 			daTF.loc[f]=Xout/Xin
			 
		# optional plot
		if plot==True:
 			fig,ax=_plt.subplots(2,sharex=True)
 			fig,ax=bodePlotFromTF(daTF,fig,ax)
 			ax[0].set_title('Bode plot of H(s)=dfOutput/dfInput')
 			_finalizeFigure(fig)
		
		return daTF
		


	
	