
import numpy as _np
import pandas as _pd
#import scipy as _sp
import matplotlib.pyplot as _plt
#import johnspythonlibraries as jpl

from johnspythonlibrary2.Plot import finalizeFigure as _finalizeFigure
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Spectral import fft_df
#from johnspythonlibrary2.Process.Misc import findNearest



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
	
	if 'float' not in dfTF.index.dtype.name:
		raise Exception('dfTF columns should be data type float.  Instead \'%s\' found'%(dfTF.index.dtype.name))
		
		
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
		y1TicksLabels=_np.array([r'$-180^o$',r'$-90^o$',r'$0^o$',r'$90^o$',r'$180^o$'])
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
	_finalizeFigure(	fig)
	
	return fig,ax
	
	

	
	
def calcTransferFunctionFromSingleTimeSeriesSpanningMultipleFrequencies(dfInput,dfOutput,plot=False):
	"""
	Calculated the transfer function from a single time series input and output signal.
	Ideally, these singals contain a freq. sweep (or similar) that has multiple frequencies contained within.
	
	Parameters
	----------
	dfInput : pandas.core.frame.DataFrame
		The original (reference) signal that goes into your system.
		Time should be index with units in seconds.
	dfOutput : pandas.core.frame.DataFrame
		The modified  signal that comes out of your system.
		Time should be index with units in seconds.
	plot : bool
		Optional plot of results
		
	Returns
	-------
	dfTF : pandas.core.frame.DataFrame
		Transfer function.  Index is frequency with units in Hz.
	
	Examples
	--------
	Example1::
		
		from johnspythonlibrary2.Process.Filters import gaussianFilter_df as _gaussianFilter_df
		from johnspythonlibrary2.Process.SigGen import chirp 

		t=_np.arange(0,20e-3,2e-6)
		fStart=2e2
		fStop=2.0e3
		y1=_chirp(t,[0.5e-3,19.46e-3],[fStart,fStop])
		df=_pd.DataFrame(y1,
					  index=t,
					  columns=['orig']) 
		fwhm=0.5e-3
		dfHP=_gaussianFilter_df(df,fwhm,'high',plot=False)
		dfHP.columns=['HP']
		dfLP=_gaussianFilter_df(df,fwhm,'low',plot=False)
		dfLP.columns=['LP']
		
		fig,ax=_plt.subplots()
		ax.plot(df)
		ax.plot(dfHP)
		ax.plot(dfLP)
	
		tf1=calcTransferFunctionFromSingleTimeSeriesSpanningMultipleFrequencies(df,dfHP)
		tf2=calcTransferFunctionFromSingleTimeSeriesSpanningMultipleFrequencies(df,dfLP)
		
		fig,ax=_plt.subplots(2,sharex=True)
		bodePlotFromTF(tf1,fig,ax)
		bodePlotFromTF(tf2,fig,ax)
		ax[0].set_xlim([0,fStop*2.])
		
	

	"""
	Xin,_,_=fft_df(dfInput)
	Xout,_,_=fft_df(dfOutput)
	
	dfTF=_pd.DataFrame(Xout.iloc[:,0]/Xin.iloc[:,0],columns=Xout.columns)
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		bodePlotFromTF(dfTF,fig,ax)
	
	return dfTF




def calcTransferFunctionFromMultipleTimeSeriesAtSingleFrequencies(	dfInput, dfOutput, plot=False):
	"""
	Calculated the transfer function from multiple input and output signals, each at a single unique frequency
	
	Parameters
	----------
	dfInput : pandas.core.frame.DataFrame
		The input signal.  Index is time with units in seconds.  Columns are the excitation frequency with units in Hz.  
	dfOutput : pandas.core.frame.DataFrame
		The output signal.  Index is time with units in seconds.  Columns are the excitation frequency with units in Hz.  
	plot : bool
		Optional plot of results
		
	Returns
	-------
	dfTF : pandas.core.frame.DataFrame
		Transfer function.  Index is frequency with units in Hz.  Single column.
	
	Examples
	--------
	Example1::
		
		yout=[]
		yin=[]
		t=_np.arange(0,10e-3,2e-6)
		freq=_np.arange(1000,12000,1000)
		for i,f in enumerate(freq):
			print(i)
			yout.append((2-i*0.1)*_np.sin(2*_np.pi*t*f+0.1*i*_np.pi))
			yin.append(2*_np.sin(2*_np.pi*t*f))
		dfInput=_pd.DataFrame(_np.array(yin).transpose(),index=t,columns=freq)
		dfOutput=_pd.DataFrame(_np.array(yout).transpose(),index=t,columns=freq)
		
		calcTransferFunctionFromMultipleTimeSeriesAtSingleFrequencies(dfInput,dfOutput,plot=True)

	"""
	from johnspythonlibrary2.Process.Spectral import fftSingleFreq_df
	
	if 'float' not in dfInput.columns.dtype.name:
		raise Exception('dfInput columns should be data type float.  Instead \'%s\' found'%(dfInput.columns.dtype.name))
		
	if 'float' not in dfOutput.columns.dtype.name:
		raise Exception('dfOutput columns should be data type float.  Instead \'%s\' found'%(dfOutput.columns.dtype.name))
	
	
	# solve input signals
	dfInTimeAve=_pd.DataFrame(index=dfInput.columns,dtype=complex)
	for i,(key,val) in enumerate(dfInput.iteritems()):
#		print(key)
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
		bodePlotFromTF(dfTF,fig,ax)
	
	return dfTF
	