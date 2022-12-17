
import numpy as _np
import pandas as _pd
import scipy as _sp
import xarray as _xr
import matplotlib.pyplot as _plt



def histogram(da, plot=False, bins=10, normalize=True, range=None):
	""" 
	Wrapper for the numpy histogram function.
	Returns a dataarray with the coords being the bin centers instead of bin edges.
	"""
	
	hist, bin_edges = _np.histogram(da.data, bins=bins, range=range)
	bins = _np.vstack((bin_edges[:-1], bin_edges[1:])).mean(axis=0)
	bins = _xr.DataArray(bins, coords={'bins': bins})
	
	if normalize is False:
		hist = _xr.DataArray(hist, coords={'bins': bins}, attrs={'long_name': 'histogram', 'units': 'count'})
	elif normalize is True:
		hist = _xr.DataArray(hist / hist.sum(), coords={'bins': bins}, attrs={'long_name': 'histogram', 'units': 'normalized'})
	
	if plot is True:
		fig, ax = _plt.subplots()
		hist.plot(ax=ax)
		
	return hist


def earth_mover_distance_1D(y1, y2, plot=False):
	"""
	
	Example
	-------
	
	::
		
		# create signal
		t = _np.arange(0, 10000) * 1e-5
		y1 = _xr.DataArray(np.sin(2 * _np.pi * 1e3 * t))
		
		# perform EMD over a range of noise values
		results = []
		amplitudes = 10 ** _np.arange(-4, 1.1, 0.1)
		_np.random.seed(0)
		for amp in amplitudes:
			y1_noisy = y1.copy() + (_np.random.rand(len(t)) - 0.5) * amp
			results.append(earth_mover_distance(y1, y1_noisy))
			
		# plot results
		fig, ax = _plt.subplots()
		ax.plot(amplitudes, results, marker='x')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel('Noise amplitude')
		ax.set_ylabel('EMD')

	Example 2 ::
		
		# create signal
		t = _np.arange(0, 10000) * 1e-5
		dt = t[1] - t[0]
		y1 = t
		y2 = t + 0.1
		print("emd = ", earth_mover_distance(y1, y2))

 
 
	"""
	y1 = y1 * 1.0
	y2 = y2 * 1.0
	from scipy.stats import wasserstein_distance as emd
	result = emd(y1, y2)
	
	if plot is True:
		if type(y1) == _xr.core.dataarray.DataArray and type(y2) == _xr.core.dataarray.DataArray:
			fig, ax = _plt.subplots(2)
			
			y1.plot(ax=ax[0], label='y1')
			y2.plot(ax=ax[0], label='y2')
			ax[0].legend()
			
			histogram(y1, bins=20).plot(ax=ax[1], label='y1')
			histogram(y2, bins=20).plot(ax=ax[1], label='y2')
			ax[1].legend()
			
			ax[0].set_title('EMD result = ' + str(result) )
	
	return result




def earth_mover_distance_v3(pdf1, pdf2):
	"""
	shared with me by jk
	
	shape of input PDFs are as follows
	[ x1, y1, ..., val_1]
	[ x2, y2, ..., val_2]
	[ .,  .,  ..., .    ]
	[ xn, yn, ..., val_n]
	
	last value is the weight (and must be non-zero)
	
	Examples
	--------
	
	Example 1 ::
		
		pdf1 = _np.array([[1, 1, 1, 1], [1, 0, 0, 1]])
		pdf2 = _np.array([[0, 0, 0, 2]])
		print(earth_mover_distance_v3(pdf1, pdf2))
		print("Should be equal to 1 + sqrt(3) = %.6f" % (1 + _np.sqrt(3)) )
		print("moving 1 block from coords (1,1,1) and another from coords (1,0,0) to (0,0,0).  Distance of first block is sqrt(3).  Distance of second block is 1.")

	"""
	# print("Work in progress.  Presently is not working... ")
	import ot
	
	r, c = _np.shape(pdf1)
	xs = pdf1[:, 0:c-1].copy(order='C')
	a = pdf1[:, c-1].copy(order='C')
	
	r, c = _np.shape(pdf2)
	xt = pdf2[:, 0:c-1].copy(order='C')
	b = pdf2[:, c-1].copy(order='C')
	
	# loss matrix
	M = ot.dist(x1=xs, x2=xt, metric='euclidean')
	norm = M.max()
	M /= norm
	
	G0 = ot.emd(a, b, M, numItermax=int(2e6))
	
	measure = _np.sum(_np.sum(_np.multiply(G0, M))) * norm
	
	return measure
	


def crossCorrelation(y1,y2,mode='same'):
	""" cross correlation wrapper for numpy function """
	CC=_np.correlate(y1,y2,mode=mode)
	return CC


def correlationCoefficient(data,fit):
	""" 
	Correlation coefficient.
	Compares a fit to data.  Note that this is only valid for a linear fit.
	
	Reference
	---------
	https://mathworld.wolfram.com/CorrelationCoefficient.html
	"""
	if type(data)==_pd.core.frame.DataFrame or type(data)==_pd.core.frame.Series:
		y=data.values.reshape(-1)
		f=fit.values.reshape(-1)
	elif type(data)==_np.ndarray:
		y=data.reshape(-1)
		f=fit.reshape(-1)
	SSxy=((f-f.mean())*(y-y.mean())).sum()
	SSxx=((f-f.mean())**2).sum()
	SSyy=((y-y.mean())**2).sum()
	rho=SSxy**2/(SSxx*SSyy)
	return rho


def errorPropagationMonteCarlo(	func,
								parameterMeanValues=[],
								parameterStdValues=[],
								N=int(1e4),
								verbose=True):
	"""
	This is the "Monte Carlo error propagation" method.  I sometimes hear it also 
	called the "Monte Carlo uncertaintly analysis" method.
	
	Parameters
	----------
	func : function
		Function to be studied.  The function takes a list of variables as input
	parameterMeanValues : list of floats
		Variables in func.
	parameterStdValues : list of floats
		Standard deviations associated with the variables in func.
	N : int
		Number of iterations to perform the Monte Carlo method.  10,000 is default
	verbose : bool
		Optionally prints the results in addition to returning them
	
	Returns
	-------
	mean : float
		The mean value of the result.  Should be approximately equal to the actual
		solved value of the function
	uncertainty : float
		The standard deviation of the result.  This is the "uncertainty" or "error"
		produced from this analysis.
	df : pandas.core.frame.DataFrame
		Dataframe containing the intermediate steps.  It's honestly not needed for
		anything other than sanity checking the steps.
	
	Examples
	--------
	def g(params):
		return 4*sp.pi**2*params[0]/params[1]**2
	mean,uncertainty,df=errorPropagationMonteCarlo( g,	[1,2],	[0.004,0.005])

	References
	----------
	http://www.eg.bucknell.edu/physics/ph310/jupyter/error_propagation_examples.ipynb.pdf
	
	"""
	
	# make sure N is an interger
	N=int(N)
	
	# initialize dataframe
	df = _pd.DataFrame(	_np.zeros((N,len(parameterMeanValues))),
						dtype=float)
	
	# populate dataframe with monte carlo uncertainties
	for i,(key,val) in enumerate(df.iteritems()):
		df[key]=_sp.random.normal(parameterMeanValues[i],parameterStdValues[i],N)

	# apply the result to each row of the dataframe
	df['result']=df.apply(func,axis=1)
	
	# calculate mean and standard deviation
	mean=df.result.mean()
	uncertainty=df.result.std()
	
	# optional output
	if verbose:
		print('Actual solved value: %.12f'%func(parameterMeanValues))
		print('Mean: %.12f'%mean )
		print('Std ("uncertainty"): %.12f'%uncertainty )
		
	return mean,uncertainty,df
	

def findNearestNeighbors(X,Y,numberOfNearestPoints=1):
	"""
	Find the nearest neighbors in X to each point in Y
	
	Example
	-------
	::
		
		from johnspythonlibraries2.Plot import finalizeSubplot as _finalizeSubplot

		# create data
		x=np.arange(0,10+1)
		y=np.arange(100,110+1)
		X,Y=np.meshgrid(x,y)
		X=X.reshape((-1,1))
		Y=Y.reshape((-1,1))
		A=np.concatenate((X,Y),axis=1)
		
		# points to investigate
		B=[[5.1,105.1],[8.9,102.55]]
		
		points,indices,radii=findNearestNeighbors(A,B,numberOfNearestPoints=5)
		
		for i in range(len(B)):
			fig,ax=plt.subplots()
			ax.plot(X,Y,'.',label='original data')
			ax.plot(B[i][0],B[i][1],'x',label='point of interest')
			ax.plot(points[i][:,0],points[i][:,1],label='nearest neighbors',marker='o',linestyle='', markerfacecolor="None")
			_finalizeSubplot(ax)
			
	"""
	
	from sklearn.neighbors import NearestNeighbors

	neigh = NearestNeighbors(n_neighbors=numberOfNearestPoints)
	neigh.fit(X)
	radii,indices=neigh.kneighbors(Y)
	points=X[indices]
	
	return points, indices, radii


def rms(data):
	"""
	Root mean square function.   Ignores NaNs.
	
	Parameters
	----------
	data : numpy.array 
		Data to be processed
		
	Return
	------
	: numpy.array 
		root mean square of data
	
	References
	----------
	# http://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
	# http://statweb.stanford.edu/~susan/courses/s60/split/node60.html
	
	Examples
	--------
	>>> rms([-1,1,-1,1,-1,1])
	1.0
	>>> rms([-1,1,-1,1,-1,1,_np.nan])
	1.0
	>>> rms([-1,1,-1,1,-1,1,_np.inf])
	Traceback (most recent call last):
	...
	Exception: data contains +- inf
	"""
	if type(data)==list:
		data=_np.array(data)
	if True in _np.isinf(data):
		raise Exception('data contains +- inf')
	return _np.sqrt(_np.nanmean((data - 0) ** 2))



	
def rejectOutliers(data, sigma=2):
	"""
	remove outliers from set of data
	
	Parameters
	----------
	data : numpy.ndarray
		data array being considered
	sigma : int
		the number of std. devs. about which to reject data.  E.g. sigma=2 
		rejects outliers outside of +-2*sigma
		
	Return
	------
	 : numpy.ndarray 
		Same as databut missing entires considered as outliers
	indicesToKeep : numpy.ndarray (of bool)
		Boolean indices associated with entries of data that are kept
	
	References
	----------
	http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
	
	Examples
	--------
	>>> rejectOutliers([1,1.1,1,1,1,10])
	(array([1. , 1.1, 1. , 1. , 1. ]), array([ True,  True,  True,  True,  True, False]))
	>>> rejectOutliers([1,1.1,1,1,1,10,_np.nan])
	(array([1. , 1.1, 1. , 1. , 1. ]), array([ True,  True,  True,  True,  True, False, False]))
	"""
	if type(data)==list:
		data=_np.array(data)
	if True in _np.isinf(data):
		raise Exception('data contains +- inf')
	indicesToKeep=_np.less(abs(data - _np.nanmean(data)), sigma* _np.nanstd(data))
	return data[indicesToKeep],indicesToKeep
				
			
if __name__ == '__main__':
    import doctest
    doctest.testmod()
