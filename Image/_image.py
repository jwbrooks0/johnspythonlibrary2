import pandas as _pd
import numpy as _np
import matplotlib.pyplot as _plt


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


def loadSequentialRawImages(partialFileName,dtype=_np.uint16,shape=(64,128),fps=None,color=True,plot=False,save=False):
	"""
	Loads sequential images.
	
	Parameters
	----------
	partialFileName : str
		Directory and filename of the images.  Use wildcards * to indicate
		unspecified text.  Example, partialFileName = 'images/*.tif'
	dtype : str or numpy data type
		You must specify the data type of the images being loaded.  'uint16' is default.
	shape : tuple with two ints
		These represent the (y,x) resolution of the figure.  NOTE that the order is (y,x).  If you get this backwards, the code will still work, but the image won't look right
	color : bool
		True - assumes that there R,B,G data associated with each image
		False - assumes greyscale.
		
	Returns
	-------
	da : xarray.core.dataarray.DataArray
		xarray DataFrame with coordinates: time,x,y,color
		where time is the image number
	
	Example
	-------
	::
		
		import numpy as np
		
		if False:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\Test1\\test3_goodCaes_15A_30sccm_29p3V'
			shape=(64,128)
			fps=390000
		elif False:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\Test1\\test4_2020_10_12_50mmlens_400kHz_15A_35p5V_15sccm'
			shape=(48,48)
			fps=400000
		elif True:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\test2\\Test1\\test2_goodCase_15A_20sccm_31p6V'
			shape=(64,128)
			fps=390000
		partialFileName='%s\\Img*.raw'%directory
		
		dtype=np.uint16
		color=True
		save=True
		da=loadSequentialRawImages(partialFileName,shape=shape,dtype=dtype,color=color,save=save,plot=True,fps=fps)
		
	References
	----------
	https://rabernat.github.io/research_computing/xarray.html
	"""
	if fps==None:
		dt=int(1)
	else:
		dt=1.0/fps

	# import libraries
	import glob
	import xarray as xr
	import numpy as np
	
	# get file names
	inList=glob.glob(partialFileName)
	dfFiles=_pd.DataFrame(inList,columns=['filepaths'],dtype=str).sort_values('filepaths').reset_index(drop=True)
	
	# initialize data storage 
	if color==True:
		video=_np.zeros((dfFiles.shape[0],shape[0],shape[1],3),dtype=dtype)
	else:
		video=_np.zeros((dfFiles.shape[0],shape[0],shape[1],1),dtype=dtype)
		
	# step through each image and import it in the data storage
	for i,(key,val) in enumerate(dfFiles.iterrows()):
# 		print(val[0]) # TODO convert to a percent complete printout
		A=_np.fromfile(val[0],dtype=dtype)
		if color==True:
			B=A.reshape((shape[0],shape[1],3))
			video[i,:,:,:]=B[::-1,:,:]
		else:
			B=A.reshape((shape[0],shape[1],1))
			video[i,:,:,0]=B[::-1,:,0] 
	
	# convert to xarray
	t=np.arange(video.shape[0])*dt
	x=np.arange(video.shape[2])
	y=np.arange(video.shape[1])
	if video.shape[3]==1:
		c=['grey']
	elif video.shape[3]==3:
		c=['blue','green','red']
	video_out=xr.DataArray(	video,
							dims=['t','y','x','color'],
							coords={	't':t,
										'x':x,
										'y':y,
										'color':c})
	
	if save==True:
		import pickle as pkl
# 		from johnspythonlibrary2.OS import processFileName
# 		baseName,dirName,fileExt,fileRoot=processFileName(val[0])
		pkl.dump(video_out,open(partialFileName.split('*')[0]+'.pkl','wb'))
 	
	if plot==True:
		_plt.figure()
		video_out[0,:,:,0].plot()

	return video_out



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


