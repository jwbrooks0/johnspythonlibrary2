
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Process.Filters import gaussianFilter_df as _gaussianFilter_df
import johnspythonlibrary2.Plot._plot as _plot
from johnspythonlibrary2.Process.Pandas import filterDFByTime as _filterDFByTime
from johnspythonlibrary2.Process.Spectral import unwrapPhase as _unwrapPhase
from johnspythonlibrary2.Process.Pandas import filterDFByTime


def leastSquareModeAnalysis(	df,
								angles,
								modeNumbers=[0,-1,-2],
								timeFWHM_phaseFilter=0.1e-3,
								plot=False,
								title=''):
	"""
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Dataframe with multiple columns associated with different angles
		index = time
	angles : numpy.ndarray
		array of angles associated with the columns in df
	mode numbers : list of ints
		mode numbers to be analyzed
	timeFWHM_phaseFilter : float
		timewidth associated with pre-frequency calculating low-pass filter
		
	Returns
	-------
	dfResults : pandas.core.frame.DataFrame
		output
	
	Examples
	--------
	::
		
		import hbtepLib as hbt
		df,dfRaw,dfMeta=hbt.get.taData_df(100000)
		angles=dfMeta.Phi.values
		dfResults=leastSquareModeAnalysis(	df*1e4,
											angles,
											[0,-1,-2])
		dfResults.plot()
	"""
	
	# initialize
	n=len(angles)
	m=len(modeNumbers)*2
	if 0 in modeNumbers:
		m-=1
		
	# construct A matrix
	A=_np.zeros((n,m))
	i=0
	for mode in modeNumbers:
		if mode == 0:
			A[:,i]=1
			i+=1
		else:
			A[:,i]=_np.sin(mode*angles)
			A[:,i+1]=_np.cos(mode*angles)
			i+=2
	Ainv=_np.linalg.pinv(A)
	
	# perform least squares analysis	
	x=Ainv.dot(df.transpose().values)
	
	# calculate amplitudes, phases, frequencies, etc.
	dfResults=_pd.DataFrame(index=df.index)
	i=0
	for mode in modeNumbers:
		if mode == 0:
			dfResults['0']=x[i,:]
			i+=1
		else:
			dfResults['%dSin'%mode]=x[i,:]
			dfResults['%dCos'%mode]=x[i+1,:]
			dfResults['%sX'%mode]=1j*dfResults['%sSin'%mode]+dfResults['%dCos'%mode]
			dfResults['%sAmp'%mode]=_np.sqrt(dfResults['%dSin'%mode]**2+dfResults['%dCos'%mode]**2)
			dfResults['%sPhase'%mode]=_np.arctan2(dfResults['%dSin'%mode],dfResults['%dCos'%mode])
			if type(timeFWHM_phaseFilter) != type(None):
				dfResults['%sPhaseFilt'%mode]=_gaussianFilter_df(	_pd.DataFrame(_unwrapPhase(dfResults['%dPhase'%mode]),index=df.index),
																	timeFWHM=timeFWHM_phaseFilter,
																	plot=False,
																	filterType='low')
			else:
				dfResults['%sPhaseFilt'%mode]=_unwrapPhase(dfResults['%dPhase'%mode])
			dfResults['%sFreq'%mode]=_np.gradient(dfResults['%sPhaseFilt'%mode])/_np.gradient(df.index.to_numpy())/(_np.pi*2)
			i+=2
			
	if plot==True:
		
		from johnspythonlibrary2.Process.Pandas import filterDFByColOrIndex
		import johnspythonlibrary2.Plot as _plot
		import matplotlib.pyplot as _plt
		from johnspythonlibrary2.Process.Misc import extractIntsFromStr
		
		fig,ax=_plt.subplots(3,sharex=True)
		
		dfTemp=filterDFByColOrIndex(dfResults,'Amp')
		for key,val in dfTemp.iteritems():
			modeNum=extractIntsFromStr(key)[0]
			ax[0].plot(val.index*1e3,val,label=modeNum)
		_plot.finalizeSubplot(ax[0],
								ylabel='',
								subtitle='amp.',
								title='%s'%title)
		
		dfTemp=filterDFByColOrIndex(dfResults,'Phase')
		dfTemp=filterDFByColOrIndex(dfTemp,'Filt',invert=True)
		for key,val in dfTemp.iteritems():
			modeNum=extractIntsFromStr(key)[0]
			ax[1].plot(val.index*1e3,val,'.',label=modeNum)
		_plot.finalizeSubplot(ax[1],
								ylabel='rad',
								subtitle='phase',
#								title='%s'%title,
								)
		
		dfTemp=filterDFByColOrIndex(dfResults,'Freq')
		for key,val in dfTemp.iteritems():
			modeNum=extractIntsFromStr(key)[0]
			ax[2].plot(val.index*1e3,val*1e-3,label=modeNum)
		_plot.finalizeSubplot(ax[2],
								ylabel='kHz',
								subtitle='Freq.',
								xlabel='Time (ms)'
								)
		
		_plot.finalizeFigure(fig)
			
			
	return dfResults

	

def removeMagneticOffsetWithCurrentReference(	dfArrayRaw,
												dfCurrent,
												timeFWHM=4e-4,
												spatialFilter=False,
												plot=False,
												shotno=0,
												y2label='G'):
	"""
	Same as standard offset subtraction but with an extra feature.  This is that
	an additional dataframe with an oscillating current is also include.  The filter
	uses the time where the current is negative to determine when to create the 
	offset.
	
	Parameters
	----------
	dfArrayRaw : pandas.core.frame.DataFrame
		Dataframe with raw signal and index of time
	dfCurrent : pandas.core.frame.DataFrame
		Dataframe with probe current and index of time
	timeFWHM : float
		time width of the Gaussian filter
	spatialFilter : bool
		optional m=n=0 filter
	plot : bool
		optional plot of results
	shotno : int
		optional title
		
	Return
	------
	dfFiltered : pandas.core.frame.DataFrame
		Dataframe with filtered output and index of time
	
	"""
	
	def filterSignalsWithProbePhasing(	dfProbe,
										dfSignal,
										plot=False,
										title='',
										timeFWHM=timeFWHM,
										shotno=0):
		
		dfSignalOld=dfSignal
		dfSignal=dfSignal.copy()
		def ranges(nums):
		    nums = sorted(set(nums))
		    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
		    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
		    return list(zip(edges, edges))
	
		# create Mask.  Only keep data wher probe current is < 0.  NAN elsewhere
		dfMask=_pd.DataFrame(dfSignal.copy().to_numpy(),columns=['Data'])
		dfMask['time']=dfSignal.copy().index
		dfMask.Data[dfProbe.iloc[:,0].to_numpy()>0]=_np.nan
		
		# add linear interpolations to fill-in NANs
		a=_np.isnan(dfMask.Data.to_numpy())
		indexNan=dfMask[a].index
		rangeNan=ranges(indexNan.to_numpy())
		for i,r in enumerate(rangeNan):
			if r[0]==0:
				continue
			if r[-1]==dfMask.shape[0]-1:
				continue
			dfMask.Data.iloc[(r[0]):(r[1]+1)]=_np.interp(dfMask.iloc[(r[0]):(r[1]+1)].time.to_numpy(),
								[dfMask.iloc[(r[0]-1)].time,dfMask.iloc[(r[1]+1)].time],
								[dfMask.iloc[(r[0]-1)].Data,dfMask.iloc[(r[1]+1)].Data])
								
		dfMask=dfMask.set_index('time')
		dfMask=dfMask.dropna()
		dfMask=_pd.DataFrame(_gaussianFilter_df(_pd.DataFrame(dfMask.Data.copy(),index=dfMask.index),timeFWHM=timeFWHM,plot=False,filterType='low'),index=dfMask.index)
		dfSignal=_filterDFByTime(dfSignal.copy(),dfMask.index[0],dfMask.index[-1])
		dfResult=_pd.DataFrame(dfSignal-dfMask.Data,index=dfMask.index.to_numpy())
		if plot==True:
			title+=', %d'%shotno
			fig,ax=_plt.subplots(3,sharex=True)
			ax[0].plot(dfProbe.index*1e3,dfProbe,label='Probe\nCurrent')
			ax[1].plot(dfSignalOld.index*1e3,dfSignalOld,label='Original')
# 			ax[1].plot(dfSignalOld.index*1e3,dfSignalOld*1e4,label='Original')
# 			ax[1].plot(dfMask.index*1e3,dfMask*1e4,label='Offset')
# 			ax[2].plot(dfResult.index*1e3,dfResult*1e4,label='Result')
			ax[1].plot(dfMask.index*1e3,dfMask,label='Offset')
			ax[2].plot(dfResult.index*1e3,dfResult,label='Result')
			_plot.finalizeSubplot(ax[0],
									   ylabel='A',
							   title=title)
			_plot.finalizeSubplot(ax[1],
									   ylabel=y2label
							   )
			_plot.finalizeSubplot(ax[2],
							   xlabel='Time (ms)',
									   ylabel=y2label
							   )
			_plot.finalizeFigure(fig,figSize=[6,3.5])
		
		return dfResult

	# clip data between 1.6e-3 and 10e-3
	dfArrayRaw=_filterDFByTime(dfArrayRaw.copy(),1.6000001e-3,10.0000001e-3)
	dfCurrent=_filterDFByTime(dfCurrent.copy(),1.6000001e-3,10.0000001e-3)
	
	# not used.  ignore
	if spatialFilter==True:
		for i,(key,val) in enumerate(dfArrayRaw.iterrows()):
			dfArrayRaw.loc[key,:]=val-val.mean()
	
	# perform filter on each signal
	dfFiltered=_pd.DataFrame(index=dfArrayRaw.index)
	for i,(key,val) in enumerate(dfArrayRaw.iteritems()):
		dfFiltered[key]=filterSignalsWithProbePhasing(dfCurrent,
													val.copy(),
													plot=plot,
													title=key,
													timeFWHM=timeFWHM,
													shotno=shotno)
		
	return dfFiltered




def offsetSubtractionWithCurrent(	dfArrayRaw,
									dfCurrent,
									timeFWHM=4e-4,
									spatialFilter=False,
									plot=False,
									title='',
									tlim=[1.1e-3,10e-3]):
	"""
	Same as standard offset subtraction but with an extra feature.  This is that
	an additional dataframe with an oscillating current is also include.  The filter
	uses the time where the current is negative to determine when to create the 
	offset.
	
	Examples
	--------
	Example1::
		
		t=_np.arange(0,10e-3,2e-6)
		yCurrent=_np.sin(2*_np.pi*t*5000)
		yCurrent[yCurrent<-0.05]=-0.05
		y1=t/5e-3
		y1[y1>1.0]=1.0
		y1+=yCurrent*0.05
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(t,yCurrent)
		ax[1].plot(t,y1)
		
		dfCurrent=_pd.DataFrame(yCurrent,index=t)
		dfArrayRaw=_pd.DataFrame(y1,index=t,columns=['asdf'])
		offsetSubtractionWithCurrent(dfArrayRaw,dfCurrent,plot=True,timeFWHM=0.2e-3,title='f1')
		
	"""
	
	def filterSignalsWithProbePhasing(dfProbe,dfSignal,plot=False,title='',
														timeFWHM=timeFWHM):
		import pandas as pd
		import numpy as np
		
		dfSignalOld=dfSignal.copy()
		dfSignal=dfSignal.copy()
		def ranges(nums):
		    nums = sorted(set(nums))
		    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
		    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
		    return list(zip(edges, edges))
	
		# create Mask.  Only keep data wher probe current is < 0.  NAN elsewhere
		dfMask=pd.DataFrame(dfSignal.copy().to_numpy(),columns=['Data'])
		dfMask['time']=dfSignal.copy().index
		dfMask.Data[dfProbe.iloc[:,0].to_numpy()>0]=np.nan
		
		# add linear interpolations to fill-in NANs
		a=np.isnan(dfMask.Data.to_numpy())
		indexNan=dfMask[a].index
		rangeNan=ranges(indexNan.to_numpy())
		for i,r in enumerate(rangeNan):
			if r[0]==0:
				continue
			if r[-1]==dfMask.shape[0]-1:
				continue
			dfMask.Data.iloc[(r[0]):(r[1]+1)]=np.interp(dfMask.iloc[(r[0]):(r[1]+1)].time.to_numpy(),
								[dfMask.iloc[(r[0]-1)].time,dfMask.iloc[(r[1]+1)].time],
								[dfMask.iloc[(r[0]-1)].Data,dfMask.iloc[(r[1]+1)].Data])
								
		dfMask=dfMask.set_index('time')
		dfMask=dfMask.dropna()
		
		dfMask=_gaussianFilter_df(dfMask.copy(),timeFWHM=timeFWHM,plot=False,filterType='low')
		
		dfSignal=filterDFByTime(dfSignal.copy(),dfMask.index[0],dfMask.index[-1])
		dfResult=pd.DataFrame(dfSignal.values-dfMask.values,index=dfMask.index.to_numpy())
		if plot==True:
			fig,ax=_plt.subplots(3,sharex=True)
			ax[0].plot(dfProbe.index*1e3,dfProbe,label='Probe\nCurrent')
			ax[1].plot(dfSignalOld.index*1e3,dfSignalOld,label='Original')
			ax[1].plot(dfMask.index*1e3,dfMask,label='Offset')
			ax[2].plot(dfResult.index*1e3,dfResult,label='Result')
			_plot.finalizeSubplot( ax[0],
								   ylabel='A',
								   title=title,
							   )
			_plot.finalizeSubplot(ax[1],
							   )
			_plot.finalizeSubplot(ax[2],
							   xlabel='Time (ms)',
							   )
			_plot.finalizeFigure(fig,figSize=[6,4.5])
		
		return dfResult

	# trim time
	dfArrayRaw=filterDFByTime(dfArrayRaw.copy(),tlim[0],tlim[1])
	dfCurrent=filterDFByTime(dfCurrent.copy(),tlim[0],tlim[1])
	
	# create empty solution Dataframe and populate it one at a time
	dfFiltered=_pd.DataFrame(index=dfArrayRaw.index)
	for i,(key,val) in enumerate(dfArrayRaw.iteritems()):
#		print(key)
		dfFiltered[key]=filterSignalsWithProbePhasing(dfCurrent,
													_pd.DataFrame(val.copy()),
													plot=plot,
													title=title+', '+str(key),
													timeFWHM=timeFWHM)
		
	return dfFiltered

