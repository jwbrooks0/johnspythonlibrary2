
import xarray as _xr
import matplotlib.pyplot as _plt
import numpy as _np
from johnspythonlibrary2.Plot import subtitle as _subTitle


def abel_inverse_transform_1D(data_in, method='three point', plot=False):
	"""
 	Wrapper for the PyAbel library

 	Parameters
 	----------
 	data_in : 2D xarray dataarray
		input data
 	method : str
		'three point', etc.
 	plot : bool, optional
		optional plot

 	Returns
 	-------
 	data_out : 2D xarray dataarray
		output data
		
 	Examples
 	--------
 	
 	Example 1::
		 
		import abel
		import numpy as np
		import matplotlib.pyplot as plt
		import xarray as xr
		
		def gauss(r, r0, sigma):
		    return np.exp(-(r-r0)**2/sigma**2)
		
		c2 = 51  # an odd-number of points
		sigma = c2/5   # 1/e-width of the Gaussian
		r = np.arange(c2)
		
		# 1-D Gaussian function ------------------
		data = gauss(r, 0, sigma)
		data_in = xr.DataArray( data, dims=['y'], coords = [r])
		
		data_out = abel_inverse_transform_1D(data_in, method='three point', plot=True)
		data_out = abel_inverse_transform_1D(data_in, method='basex', plot=True)
		
 	"""
	import abel
	
	if _np.mod(len(data_in),2)==0:
		raise Exception('Data length needs to be an odd number')
	
	# calculate step
	y=data_in.coords[data_in.dims[0]].data
	dr=_np.mean(y[1:]-y[:-1])
	
	# perform abel transform
	if method == 'three point':
		trans = abel.dasch.three_point_transform(data_in, dr=dr)
	elif method == 'basex':
		trans = abel.basex.basex_transform(data_in, dr=dr, verbose=True, basis_dir=None, direction='inverse')     
	else:
		raise Exception('Method not implemented or not recognized')
		
	data_out = _xr.DataArray( trans, dims = 'r', coords = [data_in.coords[data_in.dims[0]].data])
	
	if plot==True:
		
		fig,ax=_plt.subplots(2,sharex=True)

		data_in.plot(ax=ax[0])
		data_out.plot(ax=ax[1])
		_subTitle( ax[0], 'Input signal')
		_subTitle( ax[1], 'Transformed signal')
			
	return data_out
		
	