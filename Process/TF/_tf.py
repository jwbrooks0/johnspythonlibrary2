
import numpy as _np
import pandas as _pd
#import scipy as _sp
import matplotlib.pyplot as _plt
#import johnspythonlibraries as jpl

from johnspythonlibrary2.Plot import finalizeFigure as _finalizeFigure
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Spectral import fft_df, ifft
#from johnspythonlibrary2.Process.Misc import findNearest



def fftSignalReconstruct(s1A,s2A,s1B,s2B=None,numModes=None,plot=False):
	"""
	Creates a transfer function between an input and output signal (i.e. tf=H(s)=FFT(s1B)/FFT(s1A)) and applies it to s2A to reconstruct s2B.  

	Parameters
	----------
	s1A : pandas.core.series.Series
		Input signal used to create the transfer fuction.  Time series.
	s2A : pandas.core.series.Series.  Time series.
		Input signal that is used to reconstruct s2B with tf
	s1B : pandas.core.series.Series.  Time series.
		Output signal used to create the transfer fuction
	s2B : pandas.core.series.Series.  Time series.
		(Optional) The actual s2B signal.  If provied, it will be plotted alongside the reconstructed s2B for comparison. The anaylsis does not use it otherwise. 
	numModes : int
		The first N number of frequencies to use with the reconstruction.
		If None, the code uses all frequencies.
	plot : bool
		True - Provides an optional plot of the results

	Returns
	-------
	s2B_recon : pandas.core.series.Series
		The reconstructed s2B signal.  Time series.
		
	Examples
	--------
	::
		
		# TODO write a few examples

	"""
	
	# calculate TF from the first half of signals 1 and 2 (i.e. s1A and s2A)
	tf=calcTF(s1A,s2A,plot=True)

	# check TF by reconstructing s2A from s1A and comparing it with the actual s2A
	if plot==True:
		fftSignalReconstructFromTF(s1A,tf,plot=plot,s2=s2A)

	def trimFreqs(df,numF=50):
		fmax=1/df.shape[0]*numF
		df=df.copy()
		df[(df.index>=fmax) | (df.index<-fmax)]=0
		return df
 
	# trim tf based on the number of modes (unique frequencies) to maintain.
	if type(numModes)!=type(None):
		tf_trimmed=trimFreqs(tf,numModes)
	
	# use tf and s2A to reconstruct s2B.  
	s2B_recon=fftSignalReconstructFromTF(s1B,tf_trimmed,plot=plot,s2=s2B)

	# (Optional) Plot
	if plot==True:
		fig,ax=_plt.subplots()
		if type(s2B)!=type(None):
			ax.plot(s2B,linewidth=0.75,label=s2A.name+" Original")
		ax.plot(s2B_recon,linewidth=1,color='limegreen',label=s2A.name+' Reconstruction')
		_finalizeSubplot(ax )
		
	return s2B_recon


def fftSignalReconstructFromTF(s1,tf,s2=None,plot=False,positiveFreqsOnly=True):
	""" 
	Reconstructs a time-base signal (s2_recon) from an input signal (s1) and a transfer function (tf) 
	
	Parameters
	----------
	s1 : pandas.core.series.Series
		Input signal.  Index is time with units in seconds
	tf : pandas.core.series.Series
		Transfer function.  Index is frequency with units in Hz.
	s2 : pandas.core.series.Series
		(Optional) Original output signal.  Include it if you want it plotted alongside the reconstructed s2 for comparison
	plot : bool
		(Optional) Plot of results
		
	Returns
	-------
	s2_recon : pandas.core.series.Series
		Output signal.  Index is time with units in seconds
	"""
	
	# take fft of s1
	X=fft_df(s1)
	
	# take ifft of tf multiplied by X (i.e. the reconstruction)
	s2_recon=ifft(_pd.DataFrame(tf.iloc[:,0].values*X.iloc[:,0].values,index=tf.index))
	s2_recon.index=s1.index # make sure s1 and s2_recon have the same time basis
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(s1,label='s1')
		if type(s2)!=type(None):
			ax[1].plot(s2,label='s2')
			print((s2.values-s2_recon.values).sum())
		ax[1].plot(s2_recon,label='s2_reconstruction')
		_finalizeSubplot(ax[0],title='signal reconstruction')
		_finalizeSubplot(ax[1])
		_finalizeFigure(fig)
		
	return s2_recon


def _dB(y,yRef=1.0):
	""" calculate dB (decibel) of any number/array """
	return 20*_np.log10(y/yRef)


def _dBInverse(dBSignal):
	""" inverse calculation of dB to standard gain ratio """
	return 10**(dBSignal/20.)


def phaseCalcFromComplexSignal(dfComplex,plot=False):
	"""
	Calculates phase from a complex signal.  
	
	Parameters
	----------
	dfComplex : pandas.core.frame.DataFrame
		Complex signal. 
	plot : bool
		Optional plot
		
	Returns
	-------
	dfPhase : pandas.core.frame.DataFrame
		Dataframe containing the phase results.
		Index will be the same as dfComplex
		
	Examples
	--------
	Example1::
		
		dfComplex=_pd.DataFrame(_np.sin(_np.linspace(0,_np.pi*2*5,1000))-_np.cos(_np.linspace(0,_np.pi*2*5,1000))*1j)
		phaseCalcFromComplexSignal(dfComplex,plot=True)
	"""
	if type(dfComplex)==_pd.core.series.Series:
		dfComplex=_pd.DataFrame(dfComplex)
	dfPhase= _pd.DataFrame(_np.arctan2(_np.imag(dfComplex),_np.real(dfComplex)).reshape(-1),index=dfComplex.index,columns=dfComplex.columns)
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(dfComplex.index,_np.real(dfComplex),label='real')
		ax[0].plot(dfComplex.index,_np.imag(dfComplex),label='imag')
		ax[1].plot(dfComplex.index,dfPhase,'.',label='phase')
		
		_finalizeSubplot(	ax[0],
							  subtitle='Input',
							  )
		_finalizeSubplot(	ax[1],
							  subtitle='Phase',
							  )
		
	return dfPhase
	

def bodePlotFromTF(	dfTF,
					fig=None,
					ax=None,
					degScaleForPhase=True,
					dBScaleForAmplitude=False,
					semilogXAxis=False):
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
	
	if not (('float' in dfTF.index.dtype.name) or ('int' in dfTF.index.dtype.name)):
		raise Exception('dfTF index should be data type float.  Instead \'%s\' found'%(dfTF.index.dtype.name))
		
	dfTF=dfTF.copy()
	dfTF=dfTF.sort_index()
		
	if fig==None:
		fig,ax=_plt.subplots(2,sharex=True)
	
	# amplitude
	amp=dfTF.abs()
	if dBScaleForAmplitude==False:
		y0Label='unitless (ratio)'
		pass
	else:
		amp=_dB(amp)
		y0Label=r'unitless (dB)'
	ax[0].plot(amp,'-',label=amp.columns[0])
	
	# phase
	phase=phaseCalcFromComplexSignal(dfTF)
	if degScaleForPhase==True:
		phase*=180/_np.pi
		y1Label='deg.'
		y1Lim=[-180,180]
		y1Ticks=_np.array([-180,-90,0,90,180])
		y1TicksLabels=[]
# 		y1TicksLabels=_np.array([-180,-90,0,90,180])
# 		y1TicksLabels=_np.array([r'$-180^o$',r'$-90^o$',r'$0^o$',r'$90^o$',r'$180^o$'])
	else:
		y1Label='rad.'
		y1Lim=[-_np.pi,_np.pi]
		y1Ticks=[-_np.pi,-_np.pi*0.5,0,_np.pi*0.5,_np.pi]
		y1TicksLabels=[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$']
		pass
	ax[1].plot(phase,'.',label=phase.columns[0])
	
	# finalize plot
	_finalizeSubplot(	ax[0],
						  ylabel=y0Label,
						  subtitle='Magnitude',
#						  legendOn=False,
						  )
	_finalizeSubplot(	ax[1],
						  ylabel=y1Label,
						  xlabel='Frequency (Hz)',
						  subtitle='Phase diff.',
						  ylim=y1Lim,
						  yticks=y1Ticks,
						  ytickLabels=y1TicksLabels,
#						  legendOn=False,
						  )
	if semilogXAxis==True:
		ax[0].set_xscale('log')
		ax[0].set_xlim([amp.index[0],amp.index[-1]])
	_finalizeFigure(	fig)
	
	return fig,ax
	
	


def calcTF(dfInput,dfOutput,singleFreqs=False,plot=False):
	"""
	Calculated the transfer function from one or more input signals
	
	Parameters
	----------
	dfInput : pandas.core.frame.DataFrame
		The original (reference) signal that goes into your system.
		Time should be index with units in seconds.
		If singleFreqs=True, each column of data should be for a unique frequency and the column name should be the frequency of dtype=float.  See examples below
	dfOutput : pandas.core.frame.DataFrame
		The modified  signal that comes out of your system.
		Time should be index with units in seconds.
		If singleFreqs=True, each column of data should be for a unique frequency and the column name should be the frequency of dtype=float.  See examples below
	singleFreqs : bool
		False - dfInput and dfOutput are single column signals spanning multiple frequency components (frequency sweep, square wave, etc)
		True - dfInput and dfOutput have multiple data columns.  Each is recorded with a fixed frequency (i.e. sin(2*pi*f*t)) and each column name should the frequency of dtype=float.  See examples below.
	plot : bool
		Optional plot of results
		
	Returns
	-------
	dfTF : pandas.core.frame.DataFrame
		Transfer function.  Index is frequency with units in Hz.
	
	Examples
	--------
	Example1::
		
		### Single frequency sweep 
		
# 		from johnspythonlibrary2.Process.Filters import gaussianFilter_df as _gaussianFilter_df
		from johnspythonlibrary2.Process.SigGen import chirp as _chirp
		from johnspythonlibrary2.Process.Filters import butterworthFilter

		t=_np.arange(0,20e-3,2e-6)
		fStart=2e2
		fStop=2.0e4
		y1=_chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
		df=_pd.DataFrame(y1,
					  index=t,
					  columns=['orig'],
					  dtype=float) 
		fwhm=0.5e-3
		f_corner=1e3
		dfHP=butterworthFilter(df,f_corner,filterType='high')
# 		dfHP=_gaussianFilter_df(df,fwhm,'high',plot=False)
		dfHP.columns=['HP']
		dfLP=butterworthFilter(df,f_corner)
# 		dfLP=_gaussianFilter_df(df,fwhm,'low',plot=False)
		dfLP.columns=['LP']
		
		fig,ax=_plt.subplots()
		ax.plot(df)
		ax.plot(dfHP)
		ax.plot(dfLP)
	
		tf1=calcTF(df,dfHP)
		tf2=calcTF(df,dfLP)
		
		tf1=tf1[(tf1.index>=fStart)&(tf1.index<=fStop)]
		tf2=tf2[(tf2.index>=fStart)&(tf2.index<=fStop)]
		fig,ax=_plt.subplots(2,sharex=True)
		bodePlotFromTF(tf1,fig,ax,semilogXAxis=True)
		bodePlotFromTF(tf2,fig,ax,semilogXAxis=True)
		ax[0].set_xlim([tf1.index[0],tf1.index[-1]])

		
	Example2::
		
		### Multiple data columns, each at a unique frequency
		
		from johnspythonlibrary2.Process.Filters import butterworthFilter
		t=_np.arange(0,20e-3,2e-6)
		freqRange=10**_np.arange(2,4.05,0.05)
		y_in=_pd.DataFrame(index=t,columns=freqRange,dtype=float)
		y_out=_pd.DataFrame(index=t,columns=freqRange,dtype=float)
		f_corner=2e3
		for f in freqRange:
			y_in[f]=_np.sin(2*_np.pi*t*f)
			y_out[f]=butterworthFilter(y_in[f],f_corner)
			
		tf=calcTF(y_in,y_out,plot=False,singleFreqs=True)
		bodePlotFromTF(tf,semilogXAxis=True)
		
	"""
	
		
	if singleFreqs==False:
		Xin=fft_df(dfInput,trimNegFreqs=False)
		Xout=fft_df(dfOutput,trimNegFreqs=False)
		
		dfTF=_pd.DataFrame(Xout.iloc[:,0]/Xin.iloc[:,0],columns=Xout.columns)
		
		if plot==True:
			fig,ax=_plt.subplots(2,sharex=True)
			bodePlotFromTF(dfTF,fig,ax)
		
		return dfTF
	
	else:
		
		if not (('float' in dfInput.columns.dtype.name) or ('int' in dfInput.columns.dtype.name)):
			raise Exception('dfInput columns should be data type float (i.e. frequency).  Instead \'%s\' found'%(dfInput.columns.dtype.name))
			
		if not (('float' in dfOutput.columns.dtype.name) or ('int' in dfOutput.columns.dtype.name)):
			raise Exception('dfOutput columns should be data type float (i.e. frequency).  Instead \'%s\' found'%(dfOutput.columns.dtype.name))
			
		
		
		from johnspythonlibrary2.Process.Spectral import fftSingleFreq_df
		
		# solve input signals
		dfInTimeAve=_pd.DataFrame(index=dfInput.columns,dtype=complex)
		for i,(key,val) in enumerate(dfInput.iteritems()):
			dfInTimeAve.at[key,'result']= fftSingleFreq_df(_pd.DataFrame(val),float(key)).at['fft',key]
			
		# solve output signals
		dfTF=_pd.DataFrame(index=dfOutput.columns,dtype=complex)
		for i,(key,val) in enumerate(dfOutput.iteritems()):
			dfTF.at[key,'result']= fftSingleFreq_df(_pd.DataFrame(val),float(key)).at['fft',key]
	
		# solve for transfer function
		dfTF/=dfInTimeAve
		
		# optional plot
		if plot==True:
			fig,ax=_plt.subplots(2,sharex=True)
			fig,ax=bodePlotFromTF(dfTF,fig,ax)
			ax[0].set_title('Bode plot of H(s)=dfOutput/dfInput')
			_finalizeFigure(fig)
		
		return dfTF
		


	
	
# def calcTFFromSingleTimeSeriesSpanningMultipleFrequencies(dfInput,dfOutput,plot=False):
# 	"""
# 	Calculated the transfer function from a single time series input and output signal.
# 	Ideally, these singals contain a freq. sweep (or similar) that has multiple frequencies contained within.
# 	
# 	Parameters
# 	----------
# 	dfInput : pandas.core.frame.DataFrame
# 		The original (reference) signal that goes into your system.
# 		Time should be index with units in seconds.
# 	dfOutput : pandas.core.frame.DataFrame
# 		The modified  signal that comes out of your system.
# 		Time should be index with units in seconds.
# 	plot : bool
# 		Optional plot of results
# 		
# 	Returns
# 	-------
# 	dfTF : pandas.core.frame.DataFrame
# 		Transfer function.  Index is frequency with units in Hz.
# 	
# 	Examples
# 	--------
# 	Example1::
# 		
# 		from johnspythonlibrary2.Process.Filters import gaussianFilter_df as _gaussianFilter_df
# 		from johnspythonlibrary2.Process.SigGen import chirp as _chirp

# 		t=_np.arange(0,20e-3,2e-6)
# 		fStart=2e2
# 		fStop=2.0e3
# 		y1=_chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
# 		df=_pd.DataFrame(y1,
# 					  index=t,
# 					  columns=['orig']) 
# 		fwhm=0.5e-3
# 		dfHP=_gaussianFilter_df(df,fwhm,'high',plot=False)
# 		dfHP.columns=['HP']
# 		dfLP=_gaussianFilter_df(df,fwhm,'low',plot=False)
# 		dfLP.columns=['LP']
# 		
# 		fig,ax=_plt.subplots()
# 		ax.plot(df)
# 		ax.plot(dfHP)
# 		ax.plot(dfLP)
# 	
# 		tf1=calcTFFromSingleTimeSeriesSpanningMultipleFrequencies(df,dfHP)
# 		tf2=calcTFFromSingleTimeSeriesSpanningMultipleFrequencies(df,dfLP)
# 		
# 		tf1=tf1[(tf1.index>=fStart)&(tf1.index<=fStop)]
# 		tf2=tf2[(tf2.index>=fStart)&(tf2.index<=fStop)]
# 		fig,ax=_plt.subplots(2,sharex=True)
# 		bodePlotFromTF(tf1,fig,ax)
# 		bodePlotFromTF(tf2,fig,ax)
# 		ax[0].set_xlim([tf1.index[0],tf1.index[-1]])
# 		
# 	

# 	"""
# 	Xin=fft_df(dfInput)
# 	Xout=fft_df(dfOutput)
# 	
# 	dfTF=_pd.DataFrame(Xout.iloc[:,0]/Xin.iloc[:,0],columns=Xout.columns)
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots(2,sharex=True)
# 		bodePlotFromTF(dfTF,fig,ax)
# 	
# 	return dfTF




# def calcTransferFunctionFromMultipleTimeSeriesAtSingleFrequencies(	dfInput, dfOutput, plot=False):
# 	"""
# 	Calculated the transfer function from multiple input and output signals, each at a single unique frequency
# 	
# 	Parameters
# 	----------
# 	dfInput : pandas.core.frame.DataFrame
# 		The input signal.  Index is time with units in seconds.  Columns are the excitation frequency with units in Hz.  
# 	dfOutput : pandas.core.frame.DataFrame
# 		The output signal.  Index is time with units in seconds.  Columns are the excitation frequency with units in Hz.  
# 	plot : bool
# 		Optional plot of results
# 		
# 	Returns
# 	-------
# 	dfTF : pandas.core.frame.DataFrame
# 		Transfer function.  Index is frequency with units in Hz.  Single column.
# 	
# 	Examples
# 	--------
# 	Example1::
# 		
# 		yout=[]
# 		yin=[]
# 		t=_np.arange(0,10e-3,2e-6)
# 		freq=_np.arange(1000,12000,1000)
# 		for i,f in enumerate(freq):
# 			print(i)
# 			yout.append((2-i*0.1)*_np.sin(2*_np.pi*t*f+0.1*i*_np.pi))
# 			yin.append(2*_np.sin(2*_np.pi*t*f))
# 		dfInput=_pd.DataFrame(_np.array(yin).transpose(),index=t,columns=freq)
# 		dfOutput=_pd.DataFrame(_np.array(yout).transpose(),index=t,columns=freq)
# 		
# 		calcTransferFunctionFromMultipleTimeSeriesAtSingleFrequencies(dfInput,dfOutput,plot=True)

# 	"""
# 	from johnspythonlibrary2.Process.Spectral import fftSingleFreq_df
# 	
# 	if 'float' not in dfInput.columns.dtype.name:
# 		raise Exception('dfInput columns should be data type float.  Instead \'%s\' found'%(dfInput.columns.dtype.name))
# 		
# 	if 'float' not in dfOutput.columns.dtype.name:
# 		raise Exception('dfOutput columns should be data type float.  Instead \'%s\' found'%(dfOutput.columns.dtype.name))
# 	
# 	
# 	# solve input signals
# 	dfInTimeAve=_pd.DataFrame(index=dfInput.columns,dtype=complex)
# 	for i,(key,val) in enumerate(dfInput.iteritems()):
# #		print(key)
# 		dfInTimeAve.at[key,'result']= fftSingleFreq_df(_pd.DataFrame(val),float(key)).at['fft',key]
# 		
# 	# solve output signals
# 	dfTF=_pd.DataFrame(index=dfOutput.columns,dtype=complex)
# 	for i,(key,val) in enumerate(dfOutput.iteritems()):
# 		dfTF.at[key,'result']= fftSingleFreq_df(_pd.DataFrame(val),float(key)).at['fft',key]

# 	# solve for transfer function
# 	dfTF/=dfInTimeAve
# 	
# 	# optional plot
# 	if plot==True:
# 		fig,ax=_plt.subplots(2,sharex=True)
# 		bodePlotFromTF(dfTF,fig,ax)
# 	
# 	return dfTF
	