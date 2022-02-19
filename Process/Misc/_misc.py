
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import xarray as _xr


def symlog(arr):
	""" 
	Converts an array to `symlog' scale.  Equivalent to the symlog scale in matplotlib
	
	# TODO distinguish between a real and complex version of this function
	
	References
	----------
	 * https://pythonmatplotlibtips.blogspot.com/2018/11/x-symlog-with-shift.html
	 
	"""

	shift=0
	
	logv_im = _np.abs(arr.imag) * (10.**shift)
	logv_im[_np.where(logv_im<1.)] = 1.
	logv_im = _np.sign(arr.imag) * _np.log10(logv_im)
	
	logv_re = _np.abs(arr.real) * (10.**shift)
	logv_re[_np.where(logv_re<1.)] = 1.
	logv_re = _np.sign(arr.real) * _np.log10(logv_re)
	
	return logv_re + logv_im * 1j




def find_zero_crossings(da, rising_or_falling=None, plot=False):
	
	## Astropy's zerocross1d but with a slight modification
	def _zerocross1d(x, y, rising_or_falling=None, getIndices=False):
	  """
	    Find the zero crossing points in 1d data.
	    
	    Find the zero crossing events in a discrete data set.
	    Linear interpolation is used to determine the actual
	    locations of the zero crossing between two data points
	    showing a change in sign. Data point which are zero
	    are counted in as zero crossings if a sign change occurs
	    across them. Note that the first and last data point will
	    not be considered whether or not they are zero. 
	    
	    Parameters
	    ----------
	    x, y : arrays
	        Ordinate and abscissa data values.
	    getIndices : boolean, optional
	        If True, also the indicies of the points preceding
	        the zero crossing event will be returned. Defeualt is
	        False.
	    
	    Returns
	    -------
	    xvals : array
	        The locations of the zero crossing events determined
	        by linear interpolation on the data.
	    indices : array, optional
	        The indices of the points preceding the zero crossing
	        events. Only returned if `getIndices` is set True.
	  """
	  
	  # Check sorting of x-values
	  if _np.any((x[1:] - x[0:-1]) <= 0.0):
	    raise Exception("The x-values must be sorted in ascending order!")
	  
	  # Indices of points *before* zero-crossing
	  indi = _np.where(y[1:]*y[0:-1] < 0.0)[0]
	  
	  # (optional) check to see if the zero crossing is rising or falling.
	  if rising_or_falling == 'rising':
		    dy = y[indi+1] - y[indi]
		    indi = indi[dy>=0]
	  elif rising_or_falling == 'falling':
		    dy = y[indi+1] - y[indi]
		    indi = indi[dy<=0]
	  
	  # Find the zero crossing by linear interpolation
	  dx = x[indi+1] - x[indi]
	  dy = y[indi+1] - y[indi]
	  zc = -y[indi] * (dx/dy) + x[indi]
	  
	  # What about the points, which are actually zero
	  zi = _np.where(y == 0.0)[0]
	  # Do nothing about the first and last point should they
	  # be zero
	  zi = zi[_np.where((zi > 0) & (zi < x.size-1))]
	  # Select those point, where zero is crossed (sign change
	  # across the point)
	  zi = zi[_np.where(y[zi-1]*y[zi+1] < 0.0)]
	  
	  # Concatenate indices
	  zzindi = _np.concatenate((indi, zi)) 
	  # Concatenate zc and locations corresponding to zi
	  zz = _np.concatenate((zc, x[zi]))
	  
	  # Sort by x-value
	  sind = _np.argsort(zz)
	  zz, zzindi = zz[sind], zzindi[sind]
	  
	  if not getIndices:
	    return zz
	  else:
	    return zz, zzindi
	  
	
	x = da.coords[da.dims[0]].values
	y = da.values
	
	xvals = _zerocross1d(x, y, rising_or_falling=rising_or_falling)
	
	if plot is True:
		fig, ax = _plt.subplots()
		ax.plot(x, _np.zeros(len(x)), linestyle='--', color='grey')
		da.plot(ax=ax, label='signal')
		ax.plot(xvals, _np.zeros(len(xvals)), linestyle='', marker='x', label='zero-crossings', color='r')
		ax.legend()
	
	return xvals


def check_dims(da, dims=['t']):
	for dim in dims:
		if dim not in da.dims:
			raise Exception('Dimension, %s, not present.  Instead, %s found' % (dim, str(da.dims)))
		

def subtract_mean_and_normalize_by_std(da, dim='t'):
	if dim not in da.dims:
		raise Exception('Dimension %s not found in da.  Instead found: %s' % (dim, str(da.dims)))
	return (da.copy() - da.mean(dim=dim).data) / da.std(dim=dim).data


def subtract_mean(da, dim='t'):
	if dim not in da.dims:
		raise Exception('Dimension %s not found in da.  Instead found: %s' % (dim, str(da.dims)))
	return da.copy() - da.mean(dim=dim).data


def normalize_by_std(da, dim='t'):
	if dim not in da.dims:
		raise Exception('Dimension %s not found in da.  Instead found: %s' % (dim, str(da.dims)))
	return da.copy() / da.std(dim=dim).data


def extractFloatsFromStr(string, pattern=r"[-+]?\d*\.\d+|\d+"):
	import re
	return re.findall(pattern, string)


def extractIntsFromStr(string):
	"""
	Extracts all integers from a string.
	
	Parameters
	----------
	string : str
		str with numbers embedded
		
	Returns
	-------
	numbers : list (of int)
		list of numbers that were within string
		
	Example
	-------
	::
		
		print(extractNumsFromStr("123HelloMy65Is23"))
	
	Notes
	-----
	Does not work with decimal points.  Integers only.
	"""
	import re
	
	# get list of numbers
	numbers = re.findall(r'\d+', string)
	
	# convert to integers
	for i in range(0, len(numbers)):
		numbers[i] = int(numbers[i])
		
	return numbers
	 

def findNearest(array, value):
	"""
	search through `array` and returns the `index` of the cell closest to the 
	`value`.   `array` should be sorted in ascending order
	
	Parameters
	----------
	array : numpy.array or pandas.core.frame.DataFrame
		data array to search through
	value : float or numpy.array
		index of nearest value in array
		
		
	Return
	------
	index : int
		index of value in array that is closest to value
	
	References
	----------
	http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
	
	Example
	-------
	
		data=_np.linspace(0,100,1234)
		index=findNearest(data,50.0)
		print(data[index])
		
		df=_pd.DataFrame(data)
		index=findNearest(df,50)
		df.iloc[index]
	"""
	if type(array) == _np.ndarray:
		index = (_np.abs(array - value)).argmin()
	elif type(array) == _pd.core.frame.DataFrame:
		index = (_np.abs(array - value)).idxmin().values
		
	# value = array[index] 
	return index 
	# return index, value # uncomment to return both the index AND the value
	
