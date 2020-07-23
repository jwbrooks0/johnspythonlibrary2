import pandas as _pd
import numpy as _np


def loadSequentialImages(partialFileName):
	"""
	Loads sequential images.
	
	Parameters
	----------
	partialFileName : str
		Directory and filename of the images.  Use wildcards * to indicate
		unspecified text.  Example, partialFileName = 'images/*.tif'
		
	Returns
	-------
	da : xarray.core.dataarray.DataArray
		xarray DataFrame with coordinates: x,y,image
		where image is the image number
	
	Example
	-------
	::
		
		directory='/run/user/1000/keybase/kbfs/private/jwbrooks0/Columbia/Research/thesisWork/hsVideo/101037/'
		partialFileName='%s*.tif'%directory
		da=loadSequentialImages(partialFileName)
		
	References
	----------
	https://rabernat.github.io/research_computing/xarray.html
	"""

	# import libraries
	from PIL import Image
	import glob
	import xarray as xr
	
	# get file names
	inList=glob.glob(partialFileName)
	dfFiles=_pd.DataFrame(inList,columns=['filepaths']).sort_values('filepaths').reset_index(drop=True)
	
	# load files
	img=_np.array(Image.open(dfFiles.filepaths.values[0]))
	image=_np.zeros((img.shape[0],img.shape[1],dfFiles.shape[0]),dtype=img.dtype)
	for i,(key,val) in enumerate(dfFiles.iterrows()):
		image[:,:,i]=_np.array(Image.open(val.values[0]))
	
	# convert data to xarray DataArray
	da = xr.DataArray(image, 
					  dims=['x', 'y','image'],
	                  coords={'x': _np.arange(0,img.shape[0],dtype=img.dtype),
					   'y': _np.arange(0,img.shape[1],dtype=img.dtype),
					   'image': _np.arange(0,dfFiles.shape[0],dtype=img.dtype)},
					   )
	
	return da



def readTiffStack(filename, fps=1):
	
	"""
	Reads a tiff stack and returns a xarray dataarray
	
	Note, this uses a less-than-ideal data loading method because it is more robust.  For example, I have needed to load int12 tiffs in the past, and other image loading functions have no idea what to do with int12. 
	
	References
	----------
	https://stackoverflow.com/questions/37722139/load-a-tiff-stack-in-a-numpy-array-with-python
	
	"""
	
	from PIL import Image
	import xarray as xr
	import numpy as np
	
	data=Image.open(filename)
	
	image=np.zeros(	(data.n_frames,
					 np.shape(data)[0],
					 np.shape(data)[1]),
					dtype=np.array(data).dtype)	
	
	for i in range(data.n_frames):
		data.seek(i)
		image[i,:,:]=np.array(data)
		
	import numpy as _np
	
	time=np.arange(0,image.shape[0])/fps
		
	image = xr.DataArray(image, 
						  dims=['time','y', 'x',],
		                  coords={'x': _np.arange(0,image.shape[2],dtype=np.int16),
						   'y': _np.arange(0,image.shape[1],dtype=np.int16),
						   'time': time},
						   )
	
	return image


