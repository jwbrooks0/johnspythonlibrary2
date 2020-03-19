
import numpy as _np
import pandas as _pd
#import matplotlib.pyplot as _plt
import scipy as _sp


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
	"""
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
	"""
	indicesToKeep=abs(data - _np.mean(data)) < sigma* _np.std(data)
	return data[indicesToKeep],indicesToKeep
				
			
