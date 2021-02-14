"""

Empirical dynamic modelling (EDM) toolkit

Work in progress

"""

# load standard libraries
import time as _time

# load 3rd party libraries
import numpy as _np
import pandas as _pd
import xarray as _xr
import matplotlib.pyplot as _plt
#from deprecated import deprecated
from multiprocessing import cpu_count as _cpu_count
from joblib import Parallel as _Parallel
from joblib import delayed  as _delayed
from scipy.stats import binned_statistic_dd as _binned_statistic_dd

# load my external libraries
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot, finalizeFigure as _finalizeFigure, subTitle as _subtitle


###################################################################################
#%% signal generation
# various generated signals to test code in this library

# load my external signal generation functions
from johnspythonlibrary2.Process.SigGen import lorentzAttractor, tentMap#, coupledHarmonicOscillator, predatorPrey, 


def twoSpeciesWithBidirectionalCausality(N,tau_d=0,IC=[0.2,0.4],plot=False,params={'Ax':3.78,'Ay':3.77,'Bxy':0.07,'Byx':0.08}):
	"""
	Coupled two equation system with bi-directional causality.  
	
	Eq. 1 in Ye 2015

	Reference
	---------
	Eq. 1 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4592974/
	
	Examples
	--------
	
	Examples 1 and 2::
	
		N=3000
		twoSpeciesWithBidirectionalCausality(N,plot=True)
		twoSpeciesWithBidirectionalCausality(N,tau_d=2,plot=True)

	"""
	
	x=_np.ones(N+tau_d,dtype=float)*IC[0]
	y=_np.ones(N+tau_d,dtype=float)*IC[1]
		
	for i in range(tau_d,N+tau_d-1):
		x[i+1]=x[i]*(params['Ax']-params['Ax']*x[i]-params['Bxy']*y[i])
		y[i+1]=y[i]*(params['Ay']-params['Ay']*y[i]-params['Byx']*x[i-tau_d])
		
	x=_xr.DataArray(	x[tau_d:],
						dims=['t'],
						coords={'t':_np.arange(_np.shape(x[tau_d:])[0])},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x.t.attrs={'units':'au'}
	y=_xr.DataArray(	y[tau_d:],
						dims=['t'],
						coords={'t':_np.arange(_np.shape(y[tau_d:])[0])},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	y.t.attrs={'units':'au'}
		
	if plot==True:
		fig,ax=_plt.subplots()
		x.plot(ax=ax,label='x')
		y.plot(ax=ax,label='y')
		ax.legend()
		
	return x,y


###################################################################################
#%% sub-functions

	
def applyForecast(s,Py,edm_map,T,plot=False):
	""" 
	The forecasting method.  Combines weights with correct indices (keys) to get the forecast 
	
	Parameters
	----------
	s : xarray.core.dataarray.DataArray
		complete signal.  first half (sx) and second half (sy) of the data
	Py : xarray.core.dataarray.DataArray
		Second half (sy) signal converted to time lagged state space
	keys : xarray.core.dataarray.DataArray
		keys (indices) of nearest neighbors
	weights: xarray.core.dataarray.DataArray
		weights of nearest neighbors
	T : int
		number of time steps in which to forecast into the future.
	plot : bool
		optional plot of results
	
	Examples
	--------
	Example 1::
		
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		import pandas as pd
		
		N=1000
		ds=lorentzAttractor(N=N)
		s=ds.x.copy()
		s['t']=np.arange(0,N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=3
		tau=1
		knn=E+1
		
		Px=convertToTimeLaggedSpace(sx,E=E,tau=tau)
		Py=convertToTimeLaggedSpace(sy,E=E,tau=tau)
		Py['t']=Py.t+N//2
		
		edm_map=createMap(Px,Py,knn=knn)
		
		results=applyForecast(s,Py,edm_map,T=10,plot=True)
	
	"""
	
	# initialize a matrix for the forecast results
	index=Py.t.values[:-T]
	results=_xr.DataArray(dims=['t','future'],
						 coords={'t':index,
								 'future':_np.arange(0,T+1)})
	
	# perform forecast
	shape=edm_map['keys'].sel(t=index).shape
	for a in results.transpose():
		y=s.sel(t=edm_map['keys'].sel(t=index).values.reshape(-1)+a.future.values).values.reshape(shape)
		results.loc[:,a.future.values] = (edm_map['weights'].sel(t=index)*y).sum(axis=1).values
		
	if plot==True:
		
		# contruct actual future data matrix to use with the pearson correlation below
		dfTActual=_xr.DataArray(	_np.zeros(results.shape),
										 dims=results.dims,
										 coords=results.coords)
		for fut in results.future.data:
			dfTActual.loc[:,fut]=Py.sel(delay=0,t=(dfTActual.t.data+fut)).data
		
		fig,ax=_plt.subplots(T+1,sharex=True,sharey=True)
		rho=_xr.DataArray(dims=['T'],
							coords={'T':_np.arange(T+1)})
		for Ti in rho.coords['T'].data:
			print(Ti)
			rho.loc[Ti]=calcCorrelationCoefficient(dfTActual.sel(future=Ti), results.sel(future=Ti))
			
		for i, Ti in enumerate(range(0,1+T)):
			dfTActual.sel(future=Ti).plot(ax=ax[i])
			results.sel(future=Ti).plot(ax=ax[i])
			ax[i].set_title('')
			ax[i].set_xlabel('')
			_subtitle(ax[i],'T=%d, rho=%.3f'%(Ti,rho.sel(T=Ti).data))
		ax[0].set_title('N=%d'%len(s))
		_finalizeFigure(fig,h_pad=0)
		
	return results
	

	
def calc_EDM_time(N,E,tau,dt=int(1)):
	""" 
	Calculates the effective "time basis" used in the time-lagged state space 
	
	Parameters
	----------
	N : int
		Number of points in the original time series data
	E : int
		Dimensionality in the time-lagged basis
	tau : int
		Time step between dimensional terms, E, used in the time-lagged basis
	dt : float
		Time step in time series data.  Default is 1.  
		
	Returns
	-------
	numpy.ndarray of floats
		The effective "time basis" used in the time-lagged state space 
	"""
	return _np.arange((E-1)*tau,N,dtype=type(dt))*dt


def calc_EMD_delay(E,tau):	
	""" 
	Calculates the effective "time basis" offsets (delay) used in the time-lagged state space 
	
	Parameters
	----------
	E : int
		Dimensionality in the time-lagged basis
	tau : int
		Time step between dimensional terms, E, used in the time-lagged basis
		
	Returns
	-------
	numpy.ndarray of ints
		the effective "time basis" offsets (delay) used in the time-lagged state space 
	"""
	return _np.arange(-(E-1)*tau,1,tau,dtype=int)



def calcWeights(radii,method='exponential'):
	""" 
	Calculates weights used with the findNearestNeighbors() function
	
	Example
	-------
	Example 1::
		
		# create data
		x=_np.arange(0,10+1)
		y=_np.arange(100,110+1)
		X,Y=_np.meshgrid(x,y)
		X=X.reshape((-1,1))
		Y=Y.reshape((-1,1))
		A=_np.concatenate((X,Y),axis=1)
		
		# points to investigate
		B=[[5.1,105.1],[8.9,102.55],[3.501,107.501]]
		
		numberOfNearestPoints=5
		points,indices,radii=findNearestNeighbors(A,B,numberOfNearestPoints=numberOfNearestPoints)
		
		weights=calcWeights(radii)
		print(weights)
		
		for i in range(len(B)):
			fig,ax=_plt.subplots()
			ax.plot(X.reshape(-1),Y.reshape(-1),'.',label='original data')
			ax.plot(B[i][0],B[i][1],'x',label='point of interest')
			ax.plot(points[i][:,0],points[i][:,1],label='%d nearest neighbors\nwith weights shown'%numberOfNearestPoints,marker='o',linestyle='', markerfacecolor="None")
			plt.legend()
	
	"""
	if type(radii) in [_np.ndarray]:
		radii=_xr.DataArray(radii)
	
	if method =='exponential':
		if True:
			weights=_np.exp(-radii/_np.array(radii.min(axis=1)).reshape(-1,1))
			weights=weights/_np.array(weights.sum(axis=1)).reshape(-1,1)
		else: # this is the old method but it threw warnings.  I've replaced it with the above, but I'm leaving the old here just in case
			weights=_np.exp(-radii/radii.min(axis=1)[:,_np.newaxis])  #  this throws the same error as below.  fix.
			weights=weights/weights.sum(axis=1)[:,_np.newaxis] 	#  this line throws a warning. fix.   "FutureWarning: Support for multi-dimensional indexing (e.g. 'obj[:,None]') is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead."
		
	elif method =='uniform':
		weights=_np.ones(radii.shape)/radii.shape[1]
	else:
		raise Exception('Incorrect weighting method provided')
	
	return _xr.DataArray(weights)



def calcCorrelationCoefficient(data,fit,plot=False):
	""" 
	Pearson correlation coefficient.
	Note that pearson correlation is rho=sqrt(r^2)=r and allows for a value from
	1 (perfectly coorelated) to 0 (no correlation) to -1 (perfectly anti-correlated)
	
	Reference
	---------
	 * Eq. 22 in https://mathworld.wolfram.com/CorrelationCoefficient.html
	
	Examples
	--------
	Example 1::
		
		## Test for positive correlation.  Simple.
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		data=np.sin(2*np.pi*f*t)
		fit=data+(np.random.rand(len(t))-0.5)*0.1
		calcCorrelationCoefficient(data,fit,plot=True)
		
	Example 3::
		
		## Test for negative correlation.  Opposite of Example 1.
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		y1=np.sin(2*np.pi*f*t)
		y2=-y1+(np.random.rand(len(t))-0.5)*0.1
		calcCorrelationCoefficient(y1,y2,plot=True)
		
	Example 4::
		
		## Test for divide by zero warning
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		data=np.sin(2*np.pi*f*t)
		fit=np.zeros(data.shape)
		calcCorrelationCoefficient(data,fit,plot=True)
		
	Example 5::
		
		## Test for divide by nan
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		data=np.sin(2*np.pi*f*t)
		fit=np.zeros(data.shape)*np.nan
		calcCorrelationCoefficient(data,fit,plot=True)
	"""
	if type(data)==_np.ndarray:
		data=_xr.DataArray(data,
						 dims=['t'],
						 coords={'t':_np.arange(data.shape[0])})  
		fit=_xr.DataArray(fit,
						 dims=['t'],
						 coords={'t':_np.arange(fit.shape[0])})
	elif type(data)==_xr.core.dataarray.DataArray:
		pass
	else:
		raise Exception('Improper data type')
		
	if True:
		
		y=data.data
		f=fit.data
		
		SSxy=((f-f.mean())*(y-y.mean())).sum()
		SSxx=((f-f.mean())**2).sum()
		SSyy=((y-y.mean())**2).sum()
		if _np.sqrt(SSxx*SSyy)!=0:
			rho=SSxy/_np.sqrt(SSxx*SSyy) # r-squared value  #TODO this line occassionally returns a RuntimeWarning.  Fix.  "RuntimeWarning: invalid value encountered in double_scalars"
			# rho[i]=SSxy**2/(SSxx*SSyy) # r-squared value
		else: # TODO possibly add a divide by nan or by inf case?
			rho=_np.nan # recently added this case for divide by zero.  i'm leaving this comment here until i'm sure the fix is bug free

		if plot==True:
			fig,ax=_plt.subplots()
			ax.plot(y,label='Original data')
			ax.plot(f,label='Reconstructed data')
			ax.legend()
			ax.set_title('Rho = %.3f'%(rho))
			
	return rho


def check_dataArray(x,resetTimeIndex=False):
	"""
	Takes in an input signal (xarray.DataArray) and makes sure it meets the various requirements used through this library.  
	This function either corrects any issues it finds or throws an error.

	Parameters
	----------
	x : Ideally numpy array or xarray.DataArray
		Input signal
	resetTimeIndex : bool
		True - resets time coordinate to [0,1,2,3,...,N-1]

	Returns
	-------
	x : xarray.core.dataarray.DataArray
		Output signal with the correct variable name for the time series data.

	Examples
	--------
	
	Example 1::
		
		# standard
		dt=0.1
		t=np.arange(1000)*dt
		y=_xr.DataArray(	_np.sin(0.00834*2*np.pi*t),
							dims=['t'],
							coords={'t':t})
		check_dataArray(y)
		
	Example 2::
		
		# numpy array
		dt=0.1
		t=np.arange(1000)*dt
		check_dataArray(_np.sin(0.00834*2*np.pi*t))
		
	Example 3::
		
		# other input types
		dt=0.1
		t=np.arange(1000)*dt
		y=_xr.DataArray(	_np.sin(0.00834*2*np.pi*t),
							dims=['t'],
							coords={'t':t})
		check_dataArray([y])
		check_dataArray(	_pd.Series(_np.sin(0.00834*2*np.pi*t)))
		
	Example 4::
		
		# time data named incorrectly
		dt=0.1
		t=np.arange(1000)*dt
		y=_xr.DataArray(	_np.sin(0.00834*2*np.pi*t),
							dims=['time'],
							coords={'time':t})
		check_dataArray(y)
		
	Example 5::
		
		# time data named incorrectly again
		dt=0.1
		t=np.arange(1000)*dt
		y=_xr.DataArray(	_np.sin(0.00834*2*np.pi*t),
							dims=['Time'],
							coords={'Time':t})
		check_dataArray(y)
		
	"""
	
	# if input is a numpy array, convert it to an xarray.DataArray structure
	if type(x) == _np.ndarray:
		x=_xr.DataArray(x,
						   dims=['t'],
							 coords={'t':_np.arange(0,_np.shape(x)[0])})
	
	# make sure data is an xarray.DataArray structure
	elif type(x) not in [_xr.core.dataarray.DataArray]: 
		raise Exception('Input should be an xarray.DataArray. Instead, %s encountered.'%(str(type(x))))
		
	# make sure time dimension is present and named correctly
	if x.coords._names==set(): # if no coordinate is present
		x=_xr.DataArray(	x,
							dims=['t'],
							coords={'t':_np.arange(x.shape[0])})
	elif 'time' in x.dims:
		x=x.rename({'time':'t'})
	elif 't' not in x.dims:
		raise Exception('time or t dimension not in signal')
		
	if resetTimeIndex==True:
		x['t']=_np.arange(x.t.shape[0]).astype(int)
	
	return x
		

def convertToTimeLaggedSpace(	s,
								E,
								tau,
								fuse=False):
	""" 
	Convert input to time lagged space using the embedded dimension, E,
	and time step, tau.
	
	Parameters
	----------
	s : xarray.DataArray or list of xarray.DataArray 
		Input signal(s) to convert to time-lagged space.  If multiple signals are provided, the signals are "fused" together.  
	E : int
		Dimensional parameter
	tau : int
		Time lag parameter
	fuse : bool
		True - fuses input signals 		
		
	Returns
	-------
	P : xarray.DataArray or list of xarray.DataArray
		Dataframe containing the input signal(s) converted to time-lagged space.  
		Signals are fused if fuse=True
	
	Example
	-------
	Example 1::
		
		s=_xr.DataArray(_np.arange(0,100),
						  dims=['t'],
						  coords={'t':_np.arange(100,200)})
		P=convertToTimeLaggedSpace(s, E=5, tau=1)
		
	Example 2::
		
		s1=_xr.DataArray(_np.arange(0,100))
		s2=_xr.DataArray(_np.arange(1000,1100))
		s=[s1,s2]
		P12=convertToTimeLaggedSpace(s, E=5, tau=1)
		
	Example 3::
		
		s=_xr.DataArray(_np.arange(0,100))
		P=convertToTimeLaggedSpace(s, E=5, tau=5)
		
	Example 4::
		
		# fusion example
		N=1000
		ds=lorentzAttractor(N=N,plot='all')
		s=[ds.x,ds.z]
		E=3
		tau=2
		fuse=True
		P=convertToTimeLaggedSpace(s, E=E, tau=tau,fuse=fuse)
			
	"""
	# make sure input is a list
	if type(s) != list:
		s=[s]
		
	# process each input
	Plist=[]	
	for si in s:
		
		# check input data
		si=check_dataArray(si)
		
		# initialize empty dataArray
		index=calc_EDM_time(si.shape[0],E=E,tau=tau,dt=1).astype(int)
		columns=calc_EMD_delay(E,tau)
		P=_xr.DataArray(dims=['t','delay'],
					 coords={'t':index,
							 'delay':columns})
		
		# populate dataarray one columns at a time.  #TODO Is there a way to do this without a for loop?
		for i,ii in enumerate(columns):
			P.loc[:,ii]=si[index+ii].values
			
		Plist.append(P)
			
	if fuse==True:
		P=_xr.concat(Plist,dim='delay')
		P['delay']=_np.arange(P.shape[1])
		Plist=[P]
		
	# return P for a single input or a list of P for multiple inputs
	if _np.shape(Plist)[0]==1:
		return Plist[0]
	else:
		return Plist
		

def createMap(PA1,PA2,knn,weightingMethod='exponential'):
	"""
	Creates an SMI map from PA1 to PA2.  
	
	Parameters
	----------
	PA1 : xarray.core.dataarray.DataArray
		Input signal, s1A, converted to time lagged state space
	PA2 : xarray.core.dataarray.DataArray
		Input signal, s2A, converted to time lagged state space
	knn : None or int
		Number of nearest neighbors.  Default (None) is E+1.
	weightingMethod : str
		'exponential' - (Default).  Exponential weighting of nearest neighbors.  
		
	Returns
	-------
	edm_map : dict with keys ('keys' and 'weights')
		edm_map['keys'] contains the keys (indices) associated with the map
		edm_map['weights'] contains the weights associated with the map
	
	Examples
	--------
	Example 1::
			
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd
		
		N=100
		s=tentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=3
		knn=E+1
		tau=2
		
		PA1,PA2=convertToTimeLaggedSpace([sx,sy],E=E,tau=tau)
		
		edm_map=createMap(PA1,PA2,knn=knn)
		
		index=N//2+5
		index=8
		fig,ax=plt.subplots()
		sx.plot(ax=ax,label='Training data',color='k')
		sy.plot(ax=ax,label='Test data',color='blue')
		plt.plot(PA2.sel(t=index).delay.values+PA2.sel(t=index).t.values+N//2+1,PA2.sel(t=index).values,'r',marker='x',label='Points in question',linewidth=2)
		for j,i in enumerate(edm_map['keys'].sel(t=index).values):
			print(j,i)
			if j==0:
				label='nearest neighbors'
			else:
				label=''
			plt.plot( 	PA1.sel(t=i).t+PA1.sel(t=i).delay+1,
						PA1.sel(t=i).values,'g',marker='.',label=label,linewidth=2)

	"""
	
	coordinates, indices, radii=findNearestNeighbors(PA1.values,PA2.values,numberOfNearestPoints=knn)
	keysOfNearestNeighbors=_xr.DataArray(indices+PA1.t[0].values,
									  dims=['t','shift'],
									  coords={'t':PA2.t.values,
											   'shift':_np.arange(knn)})
	
	radii[radii==0]=_np.nan # temporary code.  if a perfect match occurs (i.e. radii=0), then an error will occur.  This should take care of that.  
	radii=_xr.DataArray(radii,
									  dims=['t','shift'],
									  coords={'t':PA2.t.values,
											   'shift':_np.arange(knn)})
	weights=calcWeights(radii,method=weightingMethod)
	
	return {'keys':keysOfNearestNeighbors,'weights':weights}
	

def findNearestNeighbors(X,Y,numberOfNearestPoints=1,plot=False):
	"""
	Find the nearest neighbors in X to each point in Y
	
	Examples
	--------
	Example 1::
		
		# create data
		a=_np.arange(0,10+1)
		b=_np.arange(100,110+1)
		A,B=_np.meshgrid(a,b)
		A=A.reshape((-1,1))
		B=B.reshape((-1,1))
		X=_np.concatenate((A,B),axis=1)
		
		# points to investigate
		Y=np.array([[5.1,105.1],[8.9,102.55],[2,107]])
		
		numberOfNearestPoints=5
		
		# one at a time
		for y in Y:
			y=y.reshape(1,-1)
			points,indices,radii=findNearestNeighbors(X,y,numberOfNearestPoints=numberOfNearestPoints,plot=True)
			
		# or all at once
		points,indices,radii=findNearestNeighbors(X,Y,numberOfNearestPoints=numberOfNearestPoints,plot=False)
		
			
	Example 2::
		
		t=np.linspace(0,0.2,200)
		y=np.sin(2*np.pi*53.5*t[0:200])
		p=y[100:103:2]
		y=_xr.DataArray(y[0:100])
		
		E=2
		tau=2
		A=convertToTimeLaggedSpace(y,E=E,tau=tau)
		offset=A.t[0].values
		A=A.values
		B=p.reshape(1,-1)
		knn=6
		points,indices,radii=findNearestNeighbors(A,B,knn)
		points=points.reshape(knn,-1)
		indices=indices.reshape(-1)+offset
		radii=radii.reshape(-1)
		
		fig,ax=plt.subplots()
		y.plot(ax=ax,marker='.')
		ax.plot(np.arange(100,103,2),p,color='tab:blue',marker='x',label='point in question')
		for i in range(radii.shape[0]):
			if i==0:
				label='nearest neighbors'
			else:
				label=''
			ax.plot( 	np.arange(indices[i],indices[i]-(E-1)*tau-1,-tau)[::-1],
						#np.arange(indices[i],indices[i]+E),
						points[i,:],
						color='tab:orange',marker='x',label=label)
		ax.legend()
		
			
	"""
	
	from sklearn.neighbors import NearestNeighbors

	neigh = NearestNeighbors(n_neighbors=numberOfNearestPoints)
	neigh.fit(X)
	radii,indices=neigh.kneighbors(Y)
	points=X[indices]
	
	# optional plot.  plots the results of only the first fit point
	if plot==True:
		i=0
		fig,ax=_plt.subplots()
		ax.plot(X[:,0],X[:,1],'.',label='original data')
		ax.plot(Y[i,0],Y[i,1],'x',label='point of interest')
		ax.plot(points[i,:,0],points[i,:,1],label='%d nearest neighbors'%numberOfNearestPoints,marker='o',linestyle='', markerfacecolor="None")
		_plt.legend()
		
	return points, indices, radii


def printTime(string,start):
	""" print time since start """
	if string!='':
		print('%s'%(string))
	print('%.3fs'%((_time.time()-start)))


def reconstruct(sx,edm_map,time_basis=None, sy=None,plot=False):
	"""
	Performs EDM reconstruction on sx using mapping information (keys,weights)
	Note that this is only SMI reconstruction if sx is not the same signal used to create the map (keys,weights).  See the SMIReconstruction function below for details.
	
	Parameters
	----------
	sx : xarray.core.dataarray.DataArray
		Input signal to use for the reconstruction
	keys : xarray.core.dataarray.DataArray
		keys (indices) generated from createMap()
	weights : xarray.core.dataarray.DataArray
		weights generated from createMap()
	time_basis : None or array
		If the sy has a non-standard time basis, you can specify it here.
		Detault = None, which assumes sx and sy are sequential datasets.  
	sx : xarray.core.dataarray.DataArray
		Optional sy signal.  If provided and plot==True, it will be plotted to be compared with sy_recon
	
	Returns
	-------
	sy_recon : xarray.core.dataarray.DataArray
		Reconstructed signal of sy
	
	Examples
	--------
	Example 1::
		
		# standard two signal example
		N=100
		s=tentMap(N=N)
		sx,sy,s=splitData(s)
		
		E=3
		knn=E+1
		tau=2
		Px,Py=convertToTimeLaggedSpace([sx,sy],E=E,tau=tau)
		
		edm_map=createMap(Px,Py,knn=knn)
		sy_recon=reconstruct(	sx,
								    edm_map,
									sy=sy,
									plot=True)
		
		
	Example 2::
		
		# standard two signal example where I've changed the time basis on the second signal
		N=100
		s=tentMap(N=N)
		sx,sy,s=splitData(s)
		sy['t']=_np.arange(N//2+10,N+10)
		
		E=3
		knn=E+1
		tau=2
		Px,Py=convertToTimeLaggedSpace([sx,sy],E=E,tau=tau)
		
		edm_map=createMap(Px,Py,knn=knn)
		sy_recon=reconstruct(	sx,
									edm_map,
									time_basis=Py.t+sy.t[0],
									sy=sy,
									plot=True)
		
		
	Example 3::
		
		# signal fusion example
		import matplotlib.pyplot as plt; plt.close('all')
		
		N=500
		ds=lorentzAttractor(N=N,removeFirstNPoints=500)
		
		sx=[ds.x[:N//2],ds.y[:N//2]]
		sy=[ds.x[N//2:],ds.y[N//2:]]
		
		E=3
		tau=2
		knn=E+1	# simplex method
		
		## convert to time-lagged space
		P1A=convertToTimeLaggedSpace(sx,E=E,tau=tau,fuse=True)
		P1B=convertToTimeLaggedSpace(sy,E=E,tau=tau,fuse=True)

		## Create map from s1A to s1B
		edm_map=createMap(P1A,P1B,knn)
		
		recon=reconstruct(	sx[0], 
								edm_map,
								time_basis=P1B.t+sy[0].t[0],
								sy=sy[0],
								plot=True)	
		# time_basis=P1B.t+sy[0].t[0]
		# sx=sx[0]
		# sy=sy[0]
		
	"""
	# check input type
	sx=check_dataArray(sx,resetTimeIndex=True)
		
	# perform reconstruction
	t=edm_map['keys'].t.values
	shape=edm_map['keys'].shape
	temp=sx.sel(t=edm_map['keys'].values.reshape(-1)).values.reshape(shape)
	sy_recon=_xr.DataArray((temp*edm_map['weights'].values).sum(axis=1),
 				 dims=['t'],
 				 coords={'t':t})
	
	if plot==True:
		fig,(ax1,ax2)=_plt.subplots(1,2,sharex=True)
		sx.plot(ax=ax1,color='k',label='x')
		sy_recon.plot(ax=ax2,color='g',label='y reconstruction')
		if type(sy)!=type(None):
			sy=check_dataArray(sy,resetTimeIndex=True)
			sy.plot(ax=ax2,color='b',label='y actual')
			rho=calcCorrelationCoefficient(sy.where(sy_recon.t==sy.t), sy_recon)
			ax2.set_title('rho=%.3f'%rho)
		ax2.legend()
		ax1.legend()
		
	# restore time basis
	if type(time_basis)==type(None):
		sy_recon['t']=edm_map['keys'].t+sx.t[-1]+1
	else:
		sy_recon['t']=time_basis
		
	return sy_recon


def splitData(s,split='half',reset_indices=True):
	""" 
	split data, s, into a first and second signal.  By default, this splits s into half. 
	
	Parameters
	----------
	s : _xr.core.dataarray.DataArray
		input signal
	split : int or str='half'
		The number of points to put into the first signal.  shape(s)-split goes into the second signal.  
		'half' is default.  splits signal in half.  also makes sure the signal has an even number of points
	reset_indices : bool
		default = True.  Resets axis=0 to an array of 0 to np.shape(s).  The various codes in this library can be tempormental if this is not done.  
	
	Returns
	-------
	sX : (same dtype as s)
		First split of the signal
	sY : (same dtype as s)
		Second split of the signal
	s : (same dtype as s)
		Typically, this is identical to the input s.  If split=='half' and shape(s) is an odd number, then the last entry of s is truncated to make it contain an even number of points
	
	Examples
	--------
	Example 1::
		
		# split data in half but input signal contains an odd number of points
		s_in=_xr.DataArray(_np.arange(100,201))
		sX,sY,s_out=splitData(s_in)
		
	Example 2::
		
		# split data unevenly.  input signal contains an odd number of points
		s_in=_xr.DataArray(_np.arange(100,201))
		sX,sY,s_out=splitData(s_in,split=10)
		
	"""
	# check input type
	s=check_dataArray(s)
		
	# reset indices
	if reset_indices==True:
		if type(s) == _xr.core.dataarray.DataArray:
			s[s.dims[0]]=_np.arange(0,s.shape[0])
		elif type(s) == _pd.core.series.Series:
			s.index=_np.arange(0,s.shape[0])
		
	if split=='half':
		# make sure s has an even number of points.  if not, truncate last point to make it contain an even number of points
		if _np.mod(s.shape[0],2)==1:
			s=s[:-1]
		N=s.shape[0]//2
	elif type(split) == int:
		N=split
	else:
		raise Exception('Improperly defined value for split.  Should be an integer.')
		
	# split s
	sX=s[0:N]
	sY=s[N:]
	
	return sX,sY,s







###################################################################################
#%% Main functions


def denoise_signal(x,y_noisy,m=100,E=2,tau=1,plot=True,y_orig=None):	
	"""
	
	Examples
	--------
	Example 1::
		
		import pandas as pd
		import matplotlib.pyplot as plt; plt.close('all')
		import numpy as np
		import xarray as xr
		
		N=100000
		dt=0.05
		ds=lorentzAttractor(N=N,dt=dt,removeMean=True,normalize=True, removeFirstNPoints=1000)
		x=ds.x
		y=ds.z
		
		np.random.seed(0)
		y_noisy=y.copy()+np.random.normal(0,1.0,size=y.shape)
		
		if False:
			fig,ax=_plt.subplots()
			y_noisy.plot(ax=ax)
			y.plot(ax=ax)
			
		E=3
		tau=2
		m=100
		y_orig=y
		denoise_signal(x,y_noisy,m=m,E=E,tau=tau,plot=True,y_orig=y_orig)
		
	"""
	
	# check input type
	x=check_dataArray(x)
	y_noisy=check_dataArray(y_noisy)
		
	# convert signals to time lagged state space
	Px,Py_noisy=convertToTimeLaggedSpace([x,y_noisy], E, tau)
		
	# create grid for matrix, M
	temp=_np.linspace(Px.min(),Px.max(),m+2)
	temp=_np.linspace(Px.min()-(temp[1]-temp[0]),Px.max()+(temp[1]-temp[0]),m+2)
	bin_edges=(temp[:-1]+temp[1:])/2
	#bin_centers=temp[1:-1]

	# add values to E-dimensioned matrix, M
	bins=[]
	sample=[]
	#dims=[]
	#coords=[]
	for i in range(E):
		bins.append(bin_edges)
		sample.append(Px[:,i].values)
		#dims.append('x%d'%(i+1))
		#coords.append(bin_centers)
	M, _, indices=_binned_statistic_dd( sample,   ## binned_statistic_dd doesn't seem to work with E>=5
													 values=Py_noisy[:,0].values, 
													 statistic='mean',
													 bins=bins, 
													 expand_binnumbers=True)
	
	# use M to filter noisy signal
	ind=[]
	for i in range(E):
		ind.append(indices[i,:]-1)
	y_filt=_xr.DataArray(M[ind],  
					 dims='t',
					 coords= [y_noisy.t.data[:-(E-1)*tau]])
	
	if plot==True:
		rho=calcCorrelationCoefficient(y_orig[:-(E-1)*tau], y_filt)
		fig,ax=_plt.subplots()
		y_noisy.plot(ax=ax,label='y+noise')
		y_orig.plot(ax=ax,label='y original')
		y_filt.plot(ax=ax,label='y filtered')
		ax.legend()
		ax.set_title('rho = %.3f'%rho)
		
	return y_filt


def forecast(s,E,T,tau=1,knn=None,plot=False,weightingMethod=None):
	"""
	Create a map of s[first half] to s[second half] and forecast up to T steps into the future.
	
	Parameters
	----------
	s : xarray.core.dataarray.DataArray
		Input signal
	E : int
		EDM dimensionality.  2 to 10 are typical values.  
	T : int
		Number of time steps into the future to forecast.  1<T<10 is typical.  
	tau : int
		EDM time step parameter.  tau>=1.  
	knn : int
		Number of nearest neighbors for SMI search.  
	plot : bool, optional
		Plot results
	weightingMethod : NoneType or str
		None defaults to "exponential" weighting
		"exponential" weighting
		
	Returns
	----------
	rho : xarray.core.dataarray.DataArray
		Pearson correlation value for each value of 0 to T.  
		
	Examples
	--------
	Example 1::
	
		N=1000
		s=tentMap(N=N)
		E=3
		T=10
		tau=1
		knn=E+1
		
		rho,results,future_actual=forecast(s,E,T,tau,knn,True)
		
	"""
	# check input
	s=check_dataArray(s)

	# initialize parameters
	N=s.shape[0]
	
	if knn == None or knn=='simplex':
		knn=E+1
	elif knn == 'smap':
		knn=s.shape[0]//2-E+1
		
	if weightingMethod==None:
		weightingMethod='exponential'
	
	print("N=%d, E=%d, T=%d, tau=%d, knn=%d, weighting=%s"%(N,E,T,tau,knn,weightingMethod))

	# prep data
	sX,sY,s=splitData(s)
	dfX,dfY=convertToTimeLaggedSpace([sX,sY],E,tau)

	# do forecast
	edm_map=createMap(dfX,dfY,knn,weightingMethod=weightingMethod)
	results=applyForecast(s,dfY,edm_map,T,plot=False)
	
	# contruct actual future data matrix to use with the pearson correlation below
	future_actual=_xr.DataArray(	_np.zeros(results.shape),
									 dims=results.dims,
									 coords=results.coords)
	for fut in results.future.data:
		future_actual.loc[:,fut]=dfY.sel(delay=0,t=(future_actual.t.data+fut)).data
	
	# calculate pearson correlation for each step into the future
	rho=_xr.DataArray(dims=['future'],
					   coords={'future':results.future})
	for fut in rho.future.data:
		rho.loc[fut]=calcCorrelationCoefficient(future_actual.sel(future=fut), results.sel(future=fut))

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		rho.plot(ax=ax,marker='.')
		_finalizeSubplot(	ax,
							xlabel='Steps into the future',
							ylabel='Correlation',
							ylim=[-0.01,1.01],
							xlim=[0,rho.future.data[-1]],
							legendOn=False,
							title="N=%d, E=%d, T=%d, tau=%d, knn=%d, weighting=%s"%(N,E,T,tau,knn,weightingMethod))
		
		fig,ax=_plt.subplots(results.future.shape[0],sharex=True)
		for i,fut in enumerate(results.future.data):
			future_actual.sel(future=fut).plot(ax=ax[i])
			results.sel(future=fut).plot(ax=ax[i])
			ax[i].set_title('')
			ax[i].set_xlabel('')
			ax[i].tick_params(axis='both', direction='in')
			_subtitle(ax=ax[i],string='Steps = %d, rho = %.3f'%(fut,rho.sel(future=fut).data))
		_finalizeFigure(fig)
		
	return rho, results, future_actual


def SMIReconstruction(	da_s1A,
						da_s1B,
						da_s2A,
						E,
						tau,
						da_s2B=None,
						knn=None,
						plot=False,
						s1Name='s1',
						s2Name='s2',
						A='A',
						B='B',
						printStepInfo=False):
	"""
	SMI reconstruction.  
	
	Parameters
	----------
	da_s1A : xarray.core.dataarray.DataArray
		signal s1A (top left)
	da_s1B : xarray.core.dataarray.DataArray
		signal s1B (top right)
	da_s2A : xarray.core.dataarray.DataArray
		signal s2A (bottom left)
	E : int
		dimensionality of time-lagged phase space
	tau : int
		time step parameter
	da_s2B : xarray.core.dataarray.DataArray
		signal s2B (bottom right)
	knn : int
		number of nearest neighbors.  None is default = E+1
	plot : bool
		(Optional) plot
	
	Returns
	-------
	sB2_recon : xarray.core.dataarray.DataArray
		Reconstructed sB2 signal
	rho : float
		Correlation value between sB2 and sB2_reconstruction.  Value is between 0 and where 1 is perfect agreement.  

	Notes
	-----
	  * This algorithm is based on https://doi.org/10.1088%2F1361-6595%2Fab0b1f
	
	Examples
	--------
	Example 1::
		
		import pandas as pd
		import matplotlib.pyplot as plt; plt.close('all')
		import numpy as np
		
		N=10000
		T=1
		ds1=lorentzAttractor(N=N,ICs={	'x0':-9.38131377,
										    'y0':-8.42655716 , 
											'z0':29.30738524},)
		t=np.linspace(0,T+T/N,N+1)
		da_s1A=ds1.x
		da_s2A=ds1.z
		
		
		ds2=lorentzAttractor(N=N,ICs={	'x0':-9.38131377/2,
										    'y0':-8.42655716/2 , 
											'z0':29.30738524/3},)
		da_s1B=ds2.x
		da_s2B=ds2.z
		
		E=4
		knn=E+1
		tau=1
		
		sB2_recon,rho=SMIReconstruction(	da_s1A,
												da_s1B,
												da_s2A,
												E, 
												tau,  
												da_s2B=da_s2B,
												plot=True,
												s1Name='Lorentz-x',
												s2Name='Lorentz-z',
												A='IC1',
												B='IC2')
		
		
	Example 2::
		
		## signal fusion case.  Use both x and y to reconstruct z
		
		import matplotlib.pyplot as plt; plt.close('all')
		
		N=2000
		ds1=lorentzAttractor(N=N)
				
		da_s1A=[ds1.x[:N//2],ds1.y[:N//2]]
		da_s1B=[ds1.x[N//2:],ds1.y[N//2:]]
		da_s2A=ds1.z[:N//2]
		da_s2B=ds1.z[N//2:]
				
		E=3
		tau=1
		
		da_s1A=ds1.x[:N//2]
		da_s1B=ds1.x[N//2:]
		da_s2A=ds1.z[:N//2]
		da_s2B=ds1.z[N//2:]
						
		sB2_recon,rho=SMIReconstruction(	da_s1A=da_s1A, 
												da_s2A=da_s2A, 
												da_s1B=da_s1B, 
												E=E, 
												tau=tau, 
												da_s2B=da_s2B, 
												plot=True,
												s1Name='x only',
												s2Name='z',
												A='IC1',
												B='IC2')	


		da_s1A=[ds1.x[:N//2],ds1.y[:N//2]]
		da_s1B=[ds1.x[N//2:],ds1.y[N//2:]]
		da_s2A=ds1.z[:N//2]
		da_s2B=ds1.z[N//2:]
				
		sB2_recon,rho=SMIReconstruction(	da_s1A=da_s1A, 
												da_s2A=da_s2A, 
												da_s1B=da_s1B, 
												E=E, 
												tau=tau, 
												da_s2B=da_s2B, 
												plot=True,
												s1Name='x and y fusion',
												s2Name='z',
												A='IC1',
												B='IC2')		
				
	"""
	
	# reset the index to integers 
	if type(da_s1A)==list:
		for da_temp in da_s1A:
			da_temp['t']=_np.arange(da_temp.t.shape[0])
	else:
		da_s1A['t']=_np.arange(da_s1A.t.shape[0])
	if type(da_s1B)==list:
		for da_temp in da_s1B:
			da_temp['t']=_np.arange(da_temp.t.shape[0])
	else:
		da_s1B['t']=_np.arange(da_s1B.t.shape[0])
	da_s2A['t']=_np.arange(da_s2A.t.shape[0])
	try:
		da_s2B['t']=_np.arange(da_s2B.t.shape[0])
	except:
		pass
		
	# define number of nearest neighbors if not previously defined
	if type(knn)==type(None):
		knn=E+1	# simplex method
		
	if printStepInfo==True:
		print("E = %d, \ttau = %d, \tknn = %d"%(E,tau,knn),end='')
		
	## convert to time-lagged space
	if type(da_s1A)==list:
		fuse=True
	else:
		fuse=False
	P1A=convertToTimeLaggedSpace(da_s1A, E, tau, fuse=fuse)
	P1B=convertToTimeLaggedSpace(da_s1B, E, tau, fuse=fuse)
	# P2A=convertToTimeLaggedSpace(da_s2A, E, tau)

	## Create map from s1A to s1B
	edm_map=createMap(P1A,P1B,knn)
	
	## apply map to s2A to get reconstructed s2Bs.s
	s2B_recon=reconstruct(da_s2A,edm_map)
	s2B_recon['t']=P1A.t
	
	## calc rho
	rho=calcCorrelationCoefficient(da_s2B[(E-1)*tau:],s2B_recon)	
	if printStepInfo==True:
		print(", \trho = %.3f"%rho)
	
	## optional plot
	if (plot==True or plot=='all') and type(da_s2B) != type(None):
	
		## sanity check map by reconstructing s1B from s1A
		if fuse==True:
			s1B_recon=reconstruct(da_s1A[0],edm_map)
		else:
			s1B_recon=reconstruct(da_s1A,edm_map)
		s1B_recon['t']=P1A.t
		
		if fuse==True:
			rho_s1B=calcCorrelationCoefficient(da_s1B[0][(E-1)*tau:],s1B_recon)
		else:
			rho_s1B=calcCorrelationCoefficient(da_s1B[(E-1)*tau:],s1B_recon)

		fig=_plt.figure()
		ax1 = _plt.subplot(221)
		ax2 = _plt.subplot(222, sharex = ax1)
		ax3 = _plt.subplot(223, sharex = ax1)
		ax4 = _plt.subplot(224, sharex = ax2)
		ax=[ax1,ax2,ax3,ax4]
		
		if fuse==True:
			da_s1A[0].plot(ax=ax[0],label='original')
			da_s1B[0].plot(ax=ax[1],label='original')
		else:
			da_s1A.plot(ax=ax[0],label='original')
			da_s1B.plot(ax=ax[1],label='original')
		s1B_recon.plot(ax=ax[1],label='recon')
		_subtitle(ax[1], 'rho=%.3f'%(rho_s1B))
		da_s2A.plot(ax=ax[2],label='original')
		da_s2B.plot(ax=ax[3],label='original')
		s2B_recon.plot(ax=ax[3],label='recon')
		_subtitle(ax[3], 'rho=%.3f'%(rho))
		ax[1].legend()
		ax[3].legend()
		_finalizeFigure(fig,figSize=[6,4])
	
	return s2B_recon,rho
	

def ccm(	s1A,
			s1B,
			s2A,
			E,
			tau,
			s2B=None,
			knn=None,
			plot=False,
			removeOffset=False):
	
	"""
	Cross correlation map
	
	Parameters
	----------
	s1A : xarray.core.dataarray.DataArray
		signal s1A (top left)
	s1B : xarray.core.dataarray.DataArray
		signal s1B (top right)
	s2A : xarray.core.dataarray.DataArray
		signal s2A (bottom left)
	E : int
		dimensionality of time-lagged phase space
	tau : int
		time step parameter
	s2B : xarray.core.dataarray.DataArray
		signal s2B (bottom right)
	knn : int
		number of nearest neighbors.  None is default = E+1
	plot : bool
		(Optional) plot
	
	Examples
	--------
	
	Example1::
		
		#TODO.  The results of this example looks identical??  Error?  Investigate.
		# lorentz equations
		N=1000
		ds=lorentzAttractor(N=N,plot=False)
		x=ds.x
		z=ds.z
		
		# add noise
		x+=_np.random.normal(0,x.std()/1,N)
		
		# prep data
		s1A=x[0:N//2]
		s1B=x[N//2:N]
		s2A=x[0:N//2]
		s2B=x[N//2:N]
		
		# call function
		E=3
		tau=1
		knn=None
		rho=ccm(s1A,s1B,s2A,s2B=s2B,E=E,tau=tau,plot=True,knn=knn)
		
	"""
	
	# check data input
	s1A=check_dataArray(s1A,resetTimeIndex=False)
	s2A=check_dataArray(s2A,resetTimeIndex=False)
	s1B=check_dataArray(s1B,resetTimeIndex=False)
	s2B=check_dataArray(s2B,resetTimeIndex=False)
	
	# define number of nearest neighbors if not previously defined
	if type(knn)==type(None):
		knn=E+1	# simplex method
		
	# remove offset
	if removeOffset==True:
		s1A=s1A.copy()-s1A.mean()
		s1B=s1B.copy()-s1B.mean()
		s2A=s2A.copy()-s2A.mean()
		s2B=s2B.copy()-s2B.mean()
		
	# convert to time-lagged space
	P1A=convertToTimeLaggedSpace(s1A, E, tau)
	P1B=convertToTimeLaggedSpace(s1B, E, tau)	
	P2A=convertToTimeLaggedSpace(s2A, E, tau)
	P2B=convertToTimeLaggedSpace(s2B, E, tau)	
	
	## A to B
	edm_map=createMap(P1A.copy(),P1B.copy(),knn=knn)
	s2B_recon=reconstruct(s2A.copy(),edm_map)
	rho_1to2=calcCorrelationCoefficient(s2B[(E-1)*tau:],s2B_recon,plot=False)	
	
 	## B to A
	edm_map=createMap(P2A.copy(),P2B.copy(),knn)
	s1B_recon=reconstruct(s1A.copy(),edm_map)
	rho_2to1=calcCorrelationCoefficient(s1B[(E-1)*tau:],s1B_recon,plot=False)	
	
	if plot==True:
		fig,ax=_plt.subplots(1,2,sharex=True,sharey=True)
		ax[1].plot(s1B[(E-1)*tau:],s1B_recon,linestyle='',marker='.')
		ax[0].plot(s2B[(E-1)*tau:],s2B_recon,linestyle='',marker='.')
		ax[0].set_aspect('equal')
		ax[1].set_aspect('equal')
		ax[0].plot([s2B.min().data,s2B.max().data],[s2B.min().data,s2B.max().data]) 
		ax[1].plot([s1B.min().data,s1B.max().data],[s1B.min().data,s1B.max().data])  
		ax[0].set_title('s1 to s2 CCM')
		ax[1].set_title('s2 to s1 CCM')
		
	return rho_1to2, rho_2to1


def SMIParameterScan(s1A,s2A,s1B,ERange,tauRange,s2B=None,plot=False,numberCPUs=_cpu_count()-1):
	"""
	Parameter scan for SMIReconstruction()
	
	Parameters
	----------
	s1A : xarray.core.dataarray.DataArray
		signal s1A (top left)
	s1B : xarray.core.dataarray.DataArray
		signal s1B (top right)
	s2A : xarray.core.dataarray.DataArray
		signal s2A (bottom left)
	ERange : numpy.ndarray of dtype int
		E values to scan through
	tauRange : numpy.ndarray of dtype int
		tau values to scan through
	s2B : xarray.core.dataarray.DataArray
		signal s2B (bottom right)
	plot : bool
		(Optional) plot
	numberCPUs : int
		Number of cpus to dedicate for this scan.  Default is the total number minus 1.  
	
	Returns
	----------
	results : xarray.core.dataarray.DataArray
		Pearson correlation results for each value of E and tau.  2D array with coordinates E and tau.  
		
	Examples
	--------
	Example 1 ::
	
		import numpy as np
		
		N=10000
		dt=0.025
		
		# solve Lorentz equations with one set of ICs
		ds_A=lorentzAttractor(N=N,dt=dt)
		s1A=ds_A.x
		s2A=ds_A.z
		
		# solve Lorentz equations with a second set of ICs
		ds_B=lorentzAttractor(	N=N,
								dt=dt,
								ICs={	'x0':-9.38131377/2,
									    'y0':-8.42655716/2 , 
										'z0':29.30738524/3})
		s1B=ds_B.x
		s2B=ds_B.z
		
		# perform reconstruction with a parameter scan of E and tau 
		ERange=np.arange(2,12+1,1)
		tauRange=np.arange(1,100+1,2)
		results=SMIParameterScan(s1A=s1A,s2A=s2A,s1B=s1B, s2B=s2B,ERange=ERange,tauRange=tauRange,plot=True)
		
		fig=_plt.gcf()
		fig.savefig("SMIReconstruction_example_results.png",dpi=150)
		
	"""
	# check input signals
	s1A=check_dataArray(s1A)
	s2A=check_dataArray(s2A)
	s1B=check_dataArray(s1B)
	if type(s2B)!= type(None):
		s2B=check_dataArray(s2B)
		
	# SMIReconstruction function, designed for parallel processing.
	def doStuff(E,tau):
		out2,rho=SMIReconstruction(	da_s1A=s1A,
										da_s2A=s2A, 
										da_s1B=s1B,
										E=E,
										tau=tau,
										da_s2B=s2B,
										plot=True)
		return E, tau, rho
	
	# Create unique list of each pair of (E,tau)
	X,Y=_np.meshgrid(ERange,tauRange)
	X=X.reshape(-1)
	Y=Y.reshape(-1)
	
	# Do SMI scan and format results in a dataarray
	results = _Parallel(n_jobs=numberCPUs)(_delayed(doStuff)(E,tau) for E,tau in zip(X,Y))  
	results = _pd.DataFrame(results,columns=['E','tau','rho'])
	results= _xr.DataArray(	results.rho.values.reshape(tauRange.shape[0],ERange.shape[0]).transpose(),
							dims=['E','tau'],
							coords={'E':ERange,'tau':tauRange})

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		results.plot(ax=ax)
		results.idxmax(dim='E').plot(ax=ax,label='max E(tau)')
		_plt.legend()
		
	return results


def determineDimensionality(s,T,tau=1,Elist=_np.arange(1,10+1),method="simplex",weightingFunction='exponential',plot=False):
	"""
	Uses forecasting to determine the dimensionality of the input data
	
	Parameters
	----------
	s : xarray.core.dataarray.DataArray
		input signal
	T : int
		Steps into the future to step
	tau : int
		time step
	Elist : numpy.ndarray
		Values of the dimensionality, E, in which to run the analysis
	method : str
		'simplex' - simplex means knn=E+1
	weightingFunction : str
		'exponentional' - default
	plot : bool
		Plots results
	
	Examples
	--------
	
	Example 1 ::
		
		N=2000
		s=tentMap(N=N)
		T=10
		tau=1
		Elist=_np.arange(1,8+1)
		determineDimensionality(s,T,tau,Elist,plot=True)
		
	"""
	# check input
	s=check_dataArray(s)
	
	# Do forecast() for each value of E and 0 to T.  
	results= _xr.DataArray(	dims=['T','E'],
							coords={'T':_np.arange(T+1),'E':Elist})
	for i,Ei in enumerate(Elist):
		knn=Ei+1
		results.loc[:,Ei]=forecast(s,Ei,T,tau,knn,False)		
	results=results.transpose('E','T')

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots(1,sharex=True)
		
		for E in results.E:
			results.sel(E=E).plot(ax=ax,label='E=%d'%E,marker='.')
		ax.set_ylabel('rho')
		ax.legend()
		ax.set_title('Forecasting results, N=%d, method=%s, tau=%d'%(len(s),method,tau))
	
	return results


#%% Under development


# def denoise_signal(x,y_noisy,m=100,E=2,tau=1,plot=True,y_orig=None):	
# 	"""
# 	
# 	Examples
# 	--------
# 	Example 1::
# 		
# 		import pandas as pd
# 		import matplotlib.pyplot as plt; plt.close('all')
# 		import numpy as np
# 		import xarray as xr
# 		
# 		N=100000
# 		dt=0.05
# 		ds=lorentzAttractor(N=N,dt=dt,removeMean=True,normalize=True, removeFirstNPoints=1000)
# 		x=ds.x
# 		y=ds.z
# 		
# 		y_noisy=y.copy()+np.random.normal(0,1.0,size=y.shape)
# 		
# 		if False:
# 			fig,ax=_plt.subplots()
# 			y_noisy.plot(ax=ax)
# 			y.plot(ax=ax)
# 			
# 		E=2
# 		tau=1
# 		m=100	
# 		y_orig=y
# 		denoise_signal(x,y_noisy,m=m,E=E,tau=tau,plot=True,y_orig=y_orig)
# 		
# 	"""
# 	
# 	print('under development. #TODO cleanup code.  #TODO optimize code.  #TODO generalize code for general inputs.  #TODO breakup code into useful subfunctions')
# 	
# 	# check input type
# 	x=check_dataArray(x)
# 	y_noisy=check_dataArray(y_noisy)

# 	dt=float((x.t[1:].values-x.t[:-1].values).mean())
# 		
# 	# convert signals to time lagged state space
# 	Px,Py_noisy=convertToTimeLaggedSpace([x,y_noisy], E, tau)
# 		
# 	# create grid
# 	bin_centers=_np.linspace(Px.min(),Px.max(),m+2)[1:-1]
# 	temp1,temp2=_np.meshgrid(bin_centers,bin_centers,indexing='ij')
# 	Mpoints=_np.concatenate((temp1.reshape(-1,1),temp2.reshape(-1,1)),axis=1) # list of each mxm grid point
# 	
# 	# for each value of x(t), find where it fits into the grid
# 	_,indices,_=findNearestNeighbors(Mpoints,Px.values,1,plot=False)
# 	indices=indices.reshape(-1)

# 	if False:
# 		unique_indices,unique_indices_inverse=_np.unique(indices,return_inverse=True)
# 		fig,ax=_plt.subplots(2)
# 		x.plot(ax=ax[0])
# 		ax[1].plot(unique_indices_inverse)
# 	
# 	# bin data
# 	M1=_xr.DataArray(_np.zeros((m,m)),coords=[bin_centers,bin_centers],
# 					dims={'p1':bin_centers,'p2':bin_centers})
# 	Mcount=M1.copy()
# 	# TODO figure out how to do this function without this for-loop
# 	for i,ii in enumerate(indices):
# 		t=Px.t[i].data
# 		M1.loc[Mpoints[ii,0],Mpoints[ii,1]]=M1.sel(p1=Mpoints[ii,0],p2=Mpoints[ii,1]).data+Py_noisy.sel(t=t,delay=0).data
# 		Mcount.loc[Mpoints[ii,0],Mpoints[ii,1]]=Mcount.sel(p1=Mpoints[ii,0],p2=Mpoints[ii,1]).data+1
# 		
# 		
# 	# average each bin
# 	M=M1/Mcount
# 	
# 	# reconstruct signal
# 	y_filt=_np.zeros(Py_noisy.shape[0])
# 	# TODO figure out how to do this function without this for-loop
# 	for i,ii in enumerate(indices):
# 		t=Px.t[i].data
# 		#print(i,ii,t)
# 		y_filt[t-1]=M.sel(p1=Mpoints[ii,0],p2=Mpoints[ii,1])
# 		
# 	y_filt=_xr.DataArray(y_filt,
# 					   dims=['t'],
# 						 coords={'t':Py_noisy.t*dt+float(x.t[0])})
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		y_noisy.plot(ax=ax,label='y+noise',linewidth=2)
# 		if type(y_orig)!=type(None):
# 			y_orig.plot(ax=ax, label='y (original)',linewidth=2)
# 			rho=calcCorrelationCoefficient(y_orig[1:], y_filt)
# 			ax.set_title('rho=%.3f'%rho)
# 		y_filt.plot(ax=ax,label='y recon.')
# 		ax.legend()
# 		x_orig=convertToTimeLaggedSpace(x,E=E,tau=tau).sel(delay=0)
# 		
# 	return y_filt
# 	


# def denoise_signal_experimental(x,y_noisy,m=100,E=2,tau=1,plot=True,y_orig=None):	
# 	"""
# 	
# 	Examples
# 	--------
# 	Example 1::
# 		
# 		import pandas as pd
# 		import matplotlib.pyplot as plt; plt.close('all')
# 		import numpy as np
# 		import xarray as xr
# 		
# 		N=100000
# 		dt=0.05
# 		ds=lorentzAttractor(N=N,dt=dt,removeMean=True,normalize=True, removeFirstNPoints=1000)
# 		x=ds.x
# 		y=ds.z
# 		
# 		y_noisy=y.copy()+np.random.normal(0,1.0,size=y.shape)
# 		
# 		if False:
# 			fig,ax=_plt.subplots()
# 			y_noisy.plot(ax=ax)
# 			y.plot(ax=ax)
# 			
# 		E=2
# 		tau=1
# 		m=100	
# 		y_orig=y
# 		denoise_signal_experimental(x,y_noisy,m=m,E=E,tau=tau,plot=True,y_orig=y_orig)
# 		
# 	"""
# 	
# 	import scipy as sp
# 	print('under development. #TODO cleanup code.  #TODO optimize code.  #TODO generalize code for general inputs.  #TODO breakup code into useful subfunctions')
# 	
# 	# check input type
# 	x=check_dataArray(x)
# 	y_noisy=check_dataArray(y_noisy)
# 		
# 	# convert signals to time lagged state space
# 	Px,Py_noisy=convertToTimeLaggedSpace([x,y_noisy], E, tau)
# 		
# 	# create grid
# 	temp=_np.linspace(Px.min(),Px.max(),m+2)
# 	temp=_np.linspace(Px.min()-(temp[1]-temp[0]),Px.max()+(temp[1]-temp[0]),m+2)
# 	bin_edges=(temp[:-1]+temp[1:])/2
# 	bin_centers=temp[1:-1]

# 	# added values to E-dimensioned matrix, M
# 	bins=[]
# 	sample=[]
# 	dims=[]
# 	coords=[]
# 	for i in range(E):
# 		bins.append(bin_edges)
# 		sample.append(Px[:,i].values)
# 		dims.append('x%d'%(i+1))
# 		coords.append(bin_centers)
# 	soln, bin_edges_1, indices=sp.stats.binned_statistic_dd( sample, 
# 															 values=Py_noisy[:,0].values, 
# 															 statistic='mean',
# 															 bins=bins, 
# 															 expand_binnumbers=True)
# 	M= _xr.DataArray( soln,
# 						dims=dims,
# 						coords=coords)
# 	
# 	# filter noisy signal
# 	ind=[]
# 	for i in range(E):
# 		ind.append(indices[i,:]-1)
# 	y_filt=_xr.DataArray(M.data[ind],  
# 					 dims='t',
# 					 coords= [y_noisy.t.data[:-(E-1)]])
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		y_noisy.plot(ax=ax,label='y+noise')
# 		y_orig.plot(ax=ax,label='y original')
# 		y_filt.plot(ax=ax,label='y filtered')
# 		ax.legend()
# 		
# 	


def eccm(s1,s2,E,tau,lagRange=_np.arange(-8,6.1,2),plot=False,s1Name='s1',s2Name='s2',title=''):
	#TODO needs docstring
	print('work in progress.  not correct yet')
	"""
	
	Examples
	--------
	
	Examples 1::
		
		# This example is from Ye2015
		
		import numpy as _np
		
		N=10000
		lagRange=(_np.arange(-8,8.1,1)).astype(int)
		E=2
		tau=1
			
		for tau_d in [0,2,4]:
			s1,s2=twoSpeciesWithBidirectionalCausality(N=N,tau_d=tau_d,plot=False,params={'Ax':3.78,'Ay':3.77,'Bxy':0.07-0.00,'Byx':0.08-0.00})
		
			s1=s1[2000:3000]#.reset_index('t',drop=True)
			s2=s2[2000:3000]#.reset_index('t',drop=True)
			
			results=eccm(s1=s1,s2=s2,E=E,tau=tau,lagRange=lagRange,plot=True,title='tau_d = %d'%tau_d, s1Name='x',s2Name='y')
		

	"""
	
	
	N=s1.shape[0]
	
	results=_pd.DataFrame(index=lagRange)
# 	tau_d=0
	for lag in lagRange:
		# print(lag)
# 		lag=-1
		
		if lag>0:
			s1_temp=s1[lag:]#.reset_index('t',drop=True)
			s2_temp=s2[:N-lag]#.reset_index('t',drop=True)
		elif lag<0:
 			s1_temp=s1[:lag]
 			s2_temp=s2[-lag:]#.reset_index('t',drop=True)
		else:
			s1_temp=s1.copy()
			s2_temp=s2.copy()
				 

		if False:
			fig,ax=_plt.subplots()
			ax.plot(s1)
			ax.plot(s2)
			ax.set_xlim([330,350])
			ax.set_title(lag)
			
		M=s1_temp.shape[0]
		s1A=s1_temp[:M//2].copy()
		s1B=s1_temp[M//2:].copy()
		s2A=s2_temp[:M//2].copy()
		s2B=s2_temp[M//2:].copy()
		
		if s1A.shape[0]>s1B.shape[0]:
			s1A=s1A[:-1]
			s2A=s2A[:-1]
		elif s1A.shape[0]<s1B.shape[0]:
			s1B=s1B[:-1]
			s2B=s2B[:-1]
		
			
# 		print(s1A.shape,s1B.shape,s2A.shape,s2B.shape)
		
# 		rho1,rho2=ccm(s1A,s1B,s2A,E,tau,s2B,E+1,False)
		rho1,rho2=ccm(s1A=s1A,s1B=s1B,s2A=s2A,E=E,tau=tau,s2B=s2B,knn=E+1,plot=False)
# 		s1=s1
# 		rho_12=ccm(s1,s2,E=E,tau=tau,plot=False)
		results.at[-lag,'12']=rho1
# 		rho_21=ccm(s2,s1,E=E,tau=tau,plot=False)
		results.at[lag,'21']=rho2
		
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(results['12'],label='%s to %s'%(s1Name,s2Name))
	# 	ax.plot(results.index.values*-1,results['12'])
		ax.plot(results['21'],label='%s to %s'%(s2Name,s1Name))
		ax.legend()
		ax.set_title(title)
		ax.set_ylabel('Pearson correlation')
		ax.set_xlabel('lag')
		
	return results
	
	

def ccmScan(s1,s2,E,tau,Nrange,plot=False,s1Name='s1',s2Name='s2'):
	print('work in progress.  needs an overhaul')
	"""

	Examples
	--------
	
	Example1::
		
# 		import matplotlib.pyplot as plt;plt.close('all')
		# Lorentz data
		Nmax=101000
		Narray=(10**_np.arange(1.8,_np.log10(Nmax-1000)+1e-4,0.025)).astype(int)
		x,y,z=solveLorentz(N=Nmax,plot=True,dt=0.05)#,IC=[-6.861764848645989/2, -0.5145288200456304 , 32.21788628404141])
		x=x[1000:]
		y=y[1000:]
		z=z[1000:]
		
# 		# add noise to Lorentz data
# 		x+=_np.random.normal(0,x.std()/15,x.shape[0])
# 		z+=_np.random.normal(0,z.std()/15,z.shape[0])
		
		# call function
		ccmScan(_pd.Series(x),_pd.Series(z),E=3,tau=2,Nrange=Narray,plot=True,s1Name='x',s2Name='z')

	"""
	
	results=_pd.DataFrame()
	for N in Nrange:
		print(N)
		
		s1A=_pd.Series(s1[0:N//2])
		s1B=_pd.Series(s1[N//2:N])
		s2A=_pd.Series(s2[0:N//2])
		s2B=_pd.Series(s2[N//2:N])
		
		
		rho_1to2,rho_2to1=ccm(s1A,s1B,s2A,s2B,E=E,tau=tau,plot=False)
		results.at[N,'rho12']=rho_1to2
		results.at[N,'rho21']=rho_2to1
# 		print(N,rho)
		
	if plot==True:
		fig,ax=_plt.subplots()
		ax.semilogx(results.rho12,label='%s to %s CCM'%(s1Name,s2Name))
		ax.semilogx(results.rho21,label='%s to %s CCM'%(s2Name,s1Name))
		ax.semilogx(Nrange,_np.ones(Nrange.shape),linestyle='--',color='grey')
		ax.set_xlabel('N')
		ax.set_ylabel('Pearson correlation')
		_finalizeSubplot(ax)

	return results






###################################################################################
#%% Examples

def example_sugihara1990():
	"""
	Forecasting example from Sugihara's 1990 paper.  See references.  
	
	References
	----------
	 * https://www.nature.com/articles/344734a0

	"""
	import matplotlib.pyplot as plt

	N=1000
	s=tentMap(N=N)
	
	#Figure 1a
	fig,ax=plt.subplots()
	s.plot(ax=ax)
	ax.set_ylim([-1,1])
	ax.set_title('Figure 1a')

	# forecasting
	E=3
	T=10
	rho,future_forecast,future_actual=forecast(s,E=E,T=T,plot=False)
	
	#Figure 1b
	fig,ax=plt.subplots()
	ax.plot([-1,0.5],[-1,0.5],linestyle='-',color='k')
	ax.plot(future_actual.sel(future=2),future_forecast.sel(future=2),linestyle='',marker='^',ms=2)
	ax.set_xlim([-1,0.5])
	ax.set_ylim([-1,0.5])
	ax.set_xlabel('Observed')
	ax.set_ylabel('Predicted (forecast)')
	ax.set_title('Figure 1b')
	
	#Figure 1c
	fig,ax=plt.subplots()
	ax.plot([-1,0.5],[-1,0.5],linestyle='-',color='k')
	ax.plot(future_actual.sel(future=5),future_forecast.sel(future=5),linestyle='',marker='^',ms=2)
	ax.set_xlim([-1,0.5])
	ax.set_ylim([-1,0.5])
	ax.set_xlabel('Observed')
	ax.set_ylabel('Predicted (forecast)')
	ax.set_title('Figure 1c')
		
	#Figure 1d
	fig,ax=plt.subplots()
	rho.plot(ax=ax,marker='s')
	ax.set_xlim([0,10])
	ax.set_ylim([0,1.1])
	ax.set_xlabel('Predicted time')
	ax.set_ylabel('Correlation coefficient (rho)')
	ax.set_title('Figure 1d')
		
	
		
def example_lorentzAttractor_reconstruction():
	

	import matplotlib.pyplot as plt
	import numpy as np
	
	if True:
		N=500
		dt=0.2
		ds=lorentzAttractor(N=N,plot=True,dt=dt,removeFirstNPoints=1200)
		ds['t']=np.arange(ds.t.shape[0])
		x=ds.x
		y=ds.y
		z=ds.z
		
		if True:
			fig=plt.gcf()
			fig.get_axes()[0].set_xlim(N//2,N)
			fig.savefig('lorentzReconstruction_figure1_time.png')
		
		if True:
			_=lorentzAttractor(N=N,plot='all',dt=dt)
			fig=plt.gcf()
			fig.savefig('lorentzReconstruction_figure1_statespace.png')
	
		s1A=x[:N//2]
		s1B=x[N//2:]
		s2A=z[:N//2]
		s2B=z[N//2:]
		
		if True:
			fig,ax=plt.subplots(2,2)
			ax[0,0].plot(np.arange(0,N//2),s1A,label='Original')
			ax[0,1].plot(np.arange(N//2,N),s1B,label='Original')#,color='tab:blue')
			ax[1,0].plot(np.arange(0,N//2),s2A,label='Original')
			ax[1,1].plot(np.arange(N//2,N),s2B,label='Original')#,color='tab:blue')
			
			_finalizeSubplot(	ax[0,0],
										subtitle='x A (first half)',
										legendOn=False,
										xlim=[0,N//2-1],
										ylabel='x',
										title='A')
			_finalizeSubplot(	ax[0,1],
										subtitle='x B (second half)',
										legendOn=False,
										xlim=[N//2,N],
										title='B')
			_finalizeSubplot(	ax[1,0],
										subtitle='z A (first half)',
										legendOn=False,
										xlabel='Time',
										xlim=[0,N//2-1],
										ylabel='z',)
			_finalizeSubplot(	ax[1,1],
										subtitle='z B (second half)',
										legendOn=False,
										xlabel='Time',
										xlim=[N//2,N])
			ax[0,1].set_yticklabels([''])
			ax[1,1].set_yticklabels([''])
			ax[0,0].set_xticklabels([''])
			ax[0,1].set_xticklabels([''])
			fig.subplots_adjust(wspace=0.05,hspace=0.05)
			fig.suptitle('Take x and z signals and split in half (A and B)')
		
			fig.savefig('lorentzReconstruction_figure2_breakupData.png')
			
			
		if True:
			
			for E, tau in zip([3,4],[1,2]):
				print(E,tau)
				
				P1A=convertToTimeLaggedSpace(s1A.copy(), E=E, tau=tau)
				P1B=convertToTimeLaggedSpace(s1B.copy(), E=E, tau=tau)
				
				edm_map = createMap(P1A,P1B,knn=4)
				
				s1B_recon=reconstruct(s1A,edm_map,plot=False,sy=s1B)
				s2B_recon=reconstruct(s2A,edm_map,plot=False,sy=s2B)
				
				fig0, ax0 = plt.subplots(2, 2, gridspec_kw={'width_ratios': [5, 1]},sharey=False)
				s1A.plot(ax=ax0[0,0],linewidth=0.5,marker='.',markersize=3)
				s2A.plot(ax=ax0[1,0],linewidth=0.5,marker='.',markersize=3)

				index=12
				ax0[0,1].plot(s1B[0:20],linewidth=0.5,marker='.',markersize=3,color='blue') #s1A.index[-1]+1+np.arange(0,index+10)
				ax0[1,1].plot(s2B[0:20],linewidth=0.5,marker='.',markersize=3,color='blue') #s1A.index[-1]+1+np.arange(0,index+10),
				
				x1=np.arange(index-E*(tau)+(tau-1),index,tau)+1
				print(x1)
				y1=s1B.loc[x1+s1B.t[0].data].values
				ax0[0,1].plot(x1,y1,linewidth=1.5,color='r',marker='o',markerfacecolor='none')
				ax0[0,1].plot(x1[-1],y1[-1],linewidth=1.5,color='r',marker='s',markerfacecolor='none',markeredgewidth=2)
				
				ax0[1,1].plot(x1[-1],s2B.loc[x1+s2B.t[0].data].values[-1],linewidth=1.5,color='r',marker='s',markerfacecolor='none',markeredgewidth=2)
				
				z=edm_map['keys'].loc[index]
				for temp in z:
					val=temp.data
					x2=np.arange(val-E*(tau)+(tau-1),val,tau)+1
					y2=s1A.loc[x2].values
					ax0[0,0].plot(x2,y2,linewidth=1.5,color='tab:orange',marker='o',markerfacecolor='none')
					ax0[0,0].plot(x2[-1],y2[-1],linewidth=1.5,color='tab:orange',marker='s',markerfacecolor='none',markeredgewidth=2)
					
					ax0[1,0].plot(x2[-1],s2A.loc[x2].values[-1],linewidth=1.5,color='tab:orange',marker='s',markerfacecolor='none',markeredgewidth=2)
					
					
				ax0[1,1].plot(x1[-1],s2B_recon.loc[x1[-1]+s2B.t[0].data],linestyle='',marker='d',color='green',markerfacecolor='none',markeredgewidth=2)
					
				_finalizeSubplot(	ax0[0,0],
											ylabel='x',
											legendOn=False,
											ylim=[-17,17])
				_finalizeSubplot(	ax0[0,1],
											legendOn=False,
											ylim=[-17,17])
				_finalizeSubplot(	ax0[1,0],
 											xlabel='time',
											ylabel='x',
											legendOn=False,
											ylim=[8,44])
				_finalizeSubplot(	ax0[1,1],
 											xlabel='time',
											legendOn=False,
											ylim=[8,44])
				_finalizeFigure(fig0, figSize=[6.5,4.1])
				fig0.savefig('lorentz_recon_example_E_%d_tau_%d.png'%(E,tau),dpi=150)
					
	
def example_sugihara2012():
	
	"""
	Reconstruction of the figure's in Sugihara's 2012 paper on CCM
	"""
	
	print('work in progress.  the plots arent in amazing agreement for some reason')
	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	
		
		
	def function(N,rx=3.8,ry=3.5,Bxy=0.02,Byx=0.1, IC=[0.2,0.4],plot=False):
		x=np.zeros(N,dtype=float)
		y=np.zeros(N,dtype=float)
		x[0]=IC[0]
		y[0]=IC[1]
		
		for i in range(0,N-1):
			x[i+1]=x[i]*(rx-rx*x[i]-Bxy*y[i])
			y[i+1]=y[i]*(ry-ry*y[i]-Byx*x[i])
			
		if plot==True:
			fig,ax=plt.subplots()
			ax.plot(x)
			ax.plot(y)
		
		return x,y
	
	
	## Figure 3A
	if True:
		results=pd.DataFrame()
		for N in np.arange(20,3501,20).astype(int):
			
			x,y=function(N,rx=3.8,ry=3.5,Bxy=0.02,Byx=0.1,IC=[0.2,0.4],plot=False)
			
			s1A=x[:N//2]
			s1B=x[N//2:]
			s2A=y[:N//2]
			s2B=y[N//2:]
			
			E=2
			tau=1
			if N==1000:
				plot=False
			else:
				plot=False
			
			rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B=s2B,E=E,tau=tau,plot=plot)
		
			results.at[N,'rho_2B']=rho_2B
			results.at[N,'rho_1B']=rho_1B
			
			
		fig,ax=plt.subplots()
		ax.plot(results.rho_2B)
		ax.plot(results.rho_1B)
		
	## Figure 3C and 3D
	if True:
		N=1000
		
		x,y=function(N,rx=3.7,ry=3.7,Bxy=0.0,Byx=0.32,IC=[0.2,0.4],plot=False)
		
		s1A=x[:N//2]
		s1B=x[N//2:]
		s2A=y[:N//2]
		s2B=y[N//2:]
		
		E=2
		tau=1
		
		rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B=s2B,E=E,tau=tau,plot=True)
		
		
	## Figure 3B
	if True:
		
		E=2
		tau=1
		
		N=400
		results=_pd.DataFrame()
		
		for Bxy in np.arange(0,0.421,0.02):
			for Byx in np.arange(0,0.41,0.02):
				#print(Bxy,Byx)
				x,y=function(N,rx=3.8,ry=3.5,Bxy=Bxy,Byx=Byx,IC=[0.2,0.4],plot=False)
								
				s1A=x[:N//2]
				s1B=x[N//2:]
				s2A=y[:N//2]
				s2B=y[N//2:]
				
				rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B=s2B,E=E,tau=tau,plot=False)
				results.at[Bxy,Byx]=rho_1B-rho_2B
		
		import xarray as xr
		da = xr.DataArray(results.values, 
				  dims=['Bxy', 'Byx'],
                  coords={'Bxy': np.arange(0,0.421,0.02),
				   'Byx': np.arange(0,0.41,0.02)})
		
		
		fig,ax=plt.subplots()
		from matplotlib import cm
		da.plot(levels=np.arange(-0.65,.66,.1),cmap=cm.Spectral_r,center=0,cbar_kwargs={'ticks': _np.arange(-1,1.01,0.1), 'label':'Rho-Rho'})
		fig.show()



def example_Ye2015():
	
	import numpy as _np
	
	N=10000
	lagRange=(_np.arange(-8,8.1,1)).astype(int)
	E=2
	tau=1
		
	for tau_d in [0,2,4]:
# 		print(tau_d)
		
		s1,s2=twoSpeciesWithBidirectionalCausality(N=N,tau_d=tau_d,plot=False,params={'Ax':3.78,'Ay':3.77,'Bxy':0.07-0.00,'Byx':0.08-0.00})
	
		s1=s1[2000:3000]#.reset_index('t',drop=True)
		s2=s2[2000:3000]#.reset_index('t',drop=True)
		
		results=eccm(	s1=s1,
						s2=s2,
						E=E,
						tau=tau,
						lagRange=lagRange,
						plot=True,
						title='tau_d = %d'%tau_d, 
						s1Name='x',
						s2Name='y')
	
