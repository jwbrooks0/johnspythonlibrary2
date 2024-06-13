
import numpy as _np
import pandas as _pd
import scipy as _sp
import xarray as _xr
import matplotlib.pyplot as _plt
import ot as _ot

def earth_mover_distance_v4(da1, da2, plot=False):
	
	if _np.any(da1.values < 0) or _np.any(da2.values < 0):
		raise Exception("Values must be positive")
	
	if False:
		def noise(M, amp):
			return (_np.random.rand(M) - 0.5) * amp
		
		def gaussian(x, sigma, x0=0):
		    """ Return the normalized Gaussian with standard deviation sigma. """
		    # c = _np.sqrt(2 * _np.pi)
		    return _np.exp(-0.5 * ((x - x0) / sigma)**2) #  / (_np.sqrt(2 * _np.pi) * _np.abs(sigma))
		
		
		x = _np.linspace(-1.25, 1.25, 1000)
		x = _xr.DataArray(x, coords={'x': x})
		sigma = 0.05
		
		noise_amp = 0.0001
		x0_array = _np.linspace(-1, 1, 75)
		x0_array = _xr.DataArray(x0_array, coords={"x0": x0_array})
		emd_array = x0_array * 0.0
		
		_plt.ioff()
		def make_plot(da1, da2, emd_result, filename):
				
			fig, ax = _plt.subplots(2, sharex=True)
			ax[0].axvline(0, ls="--", color="grey", lw=0.5)
			ax[1].axvline(0, ls="--", color="grey", lw=0.5)
			da1.plot(ax=ax[0], label="Distribution 1")
			da2.plot(ax=ax[0], label="Distribution 2")
			ax[0].legend()
			ax[0].set_ylabel("Gaussian distributions")
			if len(emd_result) > 1:
				emd_result.plot(ax=ax[1])
			ax[0].set_ylim([0, 1.5])
			ax[1].set_ylim([0, 1.1])
			ax[0].set_xlim([-1.25, 1.25])
			ax[0].set_title("")
			ax[1].set_xlabel("x")
			ax[1].set_ylabel("EMD result")
			fig.set_tight_layout(True)
			fig.savefig(filename, dpi=100)
			_plt.close(fig)
			
		for i, x0 in enumerate(x0_array):
			da1 = gaussian(x, sigma=sigma, x0=0) + _np.abs(noise(len(x), noise_amp))
			da2 = gaussian(x, sigma=sigma, x0=x0) + _np.abs(noise(len(x), noise_amp))
			emd_array[i] = earth_mover_distance_v4(da1, da2)
			make_plot(da1, da2, emd_result=emd_array[:(i+1)], filename="%.4d.png" % i)
				
		import imageio
		from glob import glob
		import os
		
		## create animation file
		files = glob("*.png")
		ims = [imageio.imread(f) for f in files]
		imageio.mimwrite("animation.gif", ims)
		
		## delete all png files
		for file in files:
			os.remove(file)
	
	
	## normalize each signal such that integral of each = 1.0
	da1 = da1 / da1.sum()
	da2 = da2 / da2.sum()
	
	## isolate coordinates
	coords1 = da1[list(da1.coords.keys())[0]].to_numpy()
	coords1 = coords1.reshape((len(coords1), 1))
	coords2 = da1[list(da1.coords.keys())[0]].to_numpy()
	coords2 = coords2.reshape((len(coords2), 1))
	
	## calculate euclidean distance between each set of coordinates
	M = _ot.dist(x1=coords1, x2=coords2, metric='euclidean') # euclidean distance from each pair of points to every other pair of points
	norm = M.max()
	M /= norm
	
	## perform EMD
	G0 = _ot.emd(a=da1.values, b=da2.values, M=M)
	emd_result = _np.sum(_np.sum(_np.multiply(G0, M))) * norm
	
	if plot is True:
		
		fig, ax = _plt.subplots()
		da1.plot(ax=ax, label="da1")
		da2.plot(ax=ax, label="da2")
		ax.legend()
		ax.set_title("EMD = %.3e" % emd_result)
	
	return emd_result


def calc_1D_histogram(x, bins=10, plot=False):
    
    if "DataArray" in str(type(x)): 
        x = x.to_numpy()
        
    H, bin_edges = _np.histogram(x, bins=bins)
    bin_centers = _np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
    H = _xr.DataArray(H.astype(int), coords={'bins': bin_centers})
    
    if plot is True:
        
        width = (bin_centers[1:] - bin_centers[0:-1]).mean() * 0.9
        height = H.to_numpy()
    
        fig, ax = _plt.subplots()
        ax.bar(x=bin_centers, height=height, width=width, color="tab:blue", label="Distribution")
        
    return H


def histogram(values_np, 
			  plot=False, 
			  num_bins=None,
			  bin_edges=None,
			  normalize=True, 
			  range=None, # lower and upper range of the bins, None=default=(a.min(), a.max())
			  ):
	""" 
	Wrapper for the numpy histogram function.
	Returns a dataarray with the coords being the bin centers instead of bin edges.
	"""
	## convert values to numpy
	values_np = _np.array(values_np)
	
	## assign bins based on num_bins or bin_edges
	if type(num_bins) is not type(None):
		if type(num_bins) is int:
			bins = num_bins
		else:
			bins = None
	elif type(bin_edges) is not type(None):
		bins = bin_edges
	else:
		bins = None
			 
	## perform histogram
	hist, bin_edges = _np.histogram(values_np, bins=bins, range=range)
	bins = _np.vstack((bin_edges[:-1], bin_edges[1:])).mean(axis=0)
	bins = _xr.DataArray(bins, coords={'bins': bins})
	
	## (optional) normalize
	if normalize is True:
		hist = hist / hist.sum()
		units = 'normalized'
	else:
		units = 'count'
	
	hist = _xr.DataArray(hist, coords={'bins': bins}, attrs={'long_name': 'histogram', 'units': units})
	
	if plot is True:
		fig, ax = _plt.subplots()
		hist.plot(ax=ax)
		
	return hist


def earth_mover_distance_1D(y1, y2, plot=False):
	"""
	
	Tests
	-----
	>>> earth_mover_distance_1D([0], [1])
	1.0
	
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
	y1 = _np.array(y1)
	y2 = _np.array(y2)
	
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
	where x, y, ... are the coordinates
	and val are the values (or weights) associated with each coordinate (and must be non-zero)
	
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
	
	References
	----------
	 * https://mathworld.wolfram.com/CorrelationCoefficient.html
	"""
	if type(data)==_pd.core.frame.DataFrame or type(data)==_pd.core.frame.Series:
		y=data.values.reshape(-1)
		f=fit.values.reshape(-1)
	elif type(data)==_np.ndarray:
		y=data.reshape(-1)
		f=fit.reshape(-1)
	elif type(data) == _xr.core.dataarray.DataArray:
		y=data.to_numpy().reshape(-1)
		f=fit.to_numpy().reshape(-1)
	else:
		raise Exception("Data type not recognized.  ")
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
	 * http://www.eg.bucknell.edu/physics/ph310/jupyter/error_propagation_examples.ipynb.pdf
	
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
	

def findNearestNeighbors(X, Y, numberOfNearestPoints=1):
	"""
	Find the nearest neighbors in X to each point in Y
	
	Tests
	-----
	
	>>> findNearestNeighbors(_np.array([[0, 0], [0, 1], [1, 0], [1,1]]), [[0.9, 0.1]], 1)
	(array([[[1, 0]]]), array([[2]], dtype=int64), array([[0.14142136]]))
	
	
	Example
	-------
	::
		
		# from johnspythonlibraries2.Plot import finalizeSubplot as _finalizeSubplot
		import numpy as np
		import matplotlib.pyplot as plt
		
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
			# _finalizeSubplot(ax)
			
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
	 * http://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
	 * http://statweb.stanford.edu/~susan/courses/s60/split/node60.html
	
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
	 * http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
	
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
