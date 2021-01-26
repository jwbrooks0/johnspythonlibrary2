
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from functools import wraps as _wraps
import pickle as _pkl
import xarray as _xr

try:
	import johnspythonlibrary2.ReadWrite._settings as _settings
	LOCALDIRTOSAVEDATA=_settings.localDirToSaveData
except ImportError:
	print('Warning: ReadWrite/_settings.py file not found.  Have you modified the ReadWrite/_settingsTemplate.py file yet?')
	LOCALDIRTOSAVEDATA=''
	
	
#################################################
# %% Pickles

def readFromPickle(filename):
	"""
	Parameters
	----------
	filename : str
		file name and path of file to be read.

	Returns
	-------
	TYPE
		Returned file, of whatever type.  

	"""
	
	return _pkl.load(open(filename,'rb'))


def writeToPickle(data,filename):
	"""
	Parameters
	----------
	data : any data type
		File to be saved
	filename : str
		file name and path to save the data

	Returns
	-------
	None.

	"""
	
	_pkl.dump(data,open(filename,"wb"))
	


#################################################
# %% Matlab data

def readMatlab73Data(filename,dataNames=None,returnRaw=False):
	"""
	Reads matlab 7.3 data

	Parameters
	----------
	filename : str
		name and path of .mat file to be read
	colNames : list of strings, optional
		list of names of the data channels in the matlab data file
	returnRaw : bool
		if True, function returns the raw data (in library format)
		
	References
	----------
	  * https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python

	"""
	import mat73
	data_temp=mat73.loadmat(filename)
	
	if returnRaw==True:
		return data_temp
	else:
		colNames=_np.array(list(data_temp.keys()),dtype=str)[:-1]
		if type(dataNames)==type(None):
			dataNames=colNames.copy()
			
		das={}
		for ic,c in enumerate(colNames):
			t=_np.arange(data_temp[c].NumPoints)*data_temp[c].XInc+data_temp[c].XOrg
			da=_xr.DataArray((data_temp[c].Data)*data_temp[c].YInc+data_temp[c].YOrg,
								dims=['t'],
								coords={'t':t},
								attrs={'units':data_temp[c].YUnits})
			
			da.t.attrs={'units':data_temp[c].XUnits}
			das[dataNames[ic]]=da
			
		return _xr.Dataset(das)
	

def convertMatToDF(filename):
	"""
	Converts a matlab .mat file to a pandas dataframe
	
	Parameters
	----------
	filename : str
		path and filename of the file to be processed
		
	Returns 
	-------
	df : pandas dataframe
		dataframe containing the same data and headers as the .mat file
	"""
	
	# libraries
	from scipy.io import loadmat

	# load .mat file into a dictionary
	matData=loadmat(filename)
	
	# get the column (header) names for each array
	names=matData['varNames']
	colNames=[]
	for i in range(len(names)):
		colNames.append(names[i][0][0])
	
	# convert the dict to a dataframe with the appropriate column names
	df=_pd.DataFrame()
	for i in colNames:
		df[i]=matData[i][:,0]
	
	return df


#################################################
# %% HBT related
def backupDFs(func,defaultDir=LOCALDIRTOSAVEDATA,debug=False):
	"""
	Decorator for functions that return one or more DataFrames.
	This decorator stores the data locally so it doesn't need to be download from 
	the server each time the data needs to be accessed. 
	
	At present, the first argument passed to func becomes part of the name of the
	locally saved data
	
	References
	----------
	https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically/30253848#30253848
	
	Notes
	-----
	
	Example
	-------
	::
	
		@_backupDFs
		def temp(shotno,t_f=10e-3,dt=2e-6,f=2e3):
			print(shotno)
			t=_np.arange(0,t_f,dt)
			y=_np.sin(2*_np.pi*t*f)
			df=_pd.DataFrame(y,index=t)
			
			return df,df,df
		
		df1,df2,df3=temp(100000,t_f=10e-3)
		df1,df2,df3=temp(100000,t_f=10e-3)
		df1,df2,df3=temp('asdf',t_f=10e-3)
		df1,df2,df3=temp(100000,t_f=10e-3,forceDownload=True)
	""" 
	
	@_wraps(func) # allows doc-string to be visible through the decorator function
	def inner1(*args, **kwargs):
		if debug:
			print(args)
			print(kwargs)
			
		name=args[0]
		if debug:
			print(name)
			
		if 'forceDownload' in kwargs:
			forceDownload=kwargs.pop('forceDownload')
		else:
			forceDownload=False
		if debug:
			print(forceDownload)
			
		partialFileName='%s_%s'%(str(name),func.__name__)
		if debug:
			print(partialFileName)
		
		try:
			if LOCALDIRTOSAVEDATA=='':
				raise Exception('Default directory not specified.  Forcing download.')
			elif forceDownload==True:
				raise Exception('forcing download')
			import glob
			inList=glob.glob('%s%s*.pkl'%(defaultDir,partialFileName))
			if len(inList)==0:
				print('data does not exist, downloading')
				raise Exception('data does not exist, downloading')
			elif len(inList)==1:
				fileName='%s%s.pkl'%(defaultDir,partialFileName)
				df=_pd.read_pickle(fileName)
				print('loaded %s from memory'%fileName)
			else:
				df=[]
				for i in range(len(inList)):
					fileName='%s%s_%d.pkl'%(defaultDir,partialFileName,i)
					df.append(_pd.read_pickle(fileName))
					print('loaded %s from memory'%fileName)
		except:
			read=func(*args, **kwargs)
			if type(read)==_pd.core.frame.DataFrame:
				if LOCALDIRTOSAVEDATA!='':
					fileName='%s%s_%s.pkl'%(defaultDir,str(name),func.__name__)
					read.to_pickle(fileName)
					print('downloading and storing %s'%fileName)
			else:
				for i in range(len(read)):
					df=read[i]
					if LOCALDIRTOSAVEDATA!='':
						fileName='%s%s_%s_%d.pkl'%(defaultDir,str(name),func.__name__,i)
						df.to_pickle(fileName)
						print('downloading and storing %s'%fileName)
			df=read
			
		return df
					
	return inner1





#################################################
# %% ODS files

def readOdsToDF(filename, sheetName='Sheet1', header=0):
	""" 
	Convert ods file to pandas dataframe 
	
	Parameters
	----------
	filename : str
		ods filename and path
	sheetName : str
		name of the sheet in the ods file.  default it 'Sheet1'
	header : int
		number of rows dedicated to the header
		#TODO this doesn't quite work.  
		
	Returns
	-------
	df : pandas.core.frame.DataFrame
		dataframe of the ods file
		
		
		
	
	"""
	import ezodf
	
	tab = ezodf.opendoc(filename=filename).sheets[sheetName]
	df = _pd.DataFrame({col[header].value:[x.value for x in col[header+1:]]
						 for col in tab.columns()})
	
	return df



#################################################
# %% HDF5

def xr_DataArray_to_hdf5(	y, 
 							h5py_file, 
							var_name='',
							path='data', 
							metadata={}):
	"""
	
	Examples
	--------
	Example 1::
	
		# multiple time series data with different time bases
		t1=np.arange(0,1e-2,1e-6)
		y1=np.sin(2*np.pi*1e3*t1)
		y1=xr.DataArray(	y1,
							dims=['t'],
							coords=[t1])
		y1.name='y1'
		
		t2=np.arange(0,1e-2,2e-6)
		y2=np.cos(2*np.pi*1e3*t2)
		y2=xr.DataArray(	y2,
							dims=['t'],
							coords=[t2])
		y2.name='y2'
		# t2=np.arange(0,1e-2,2e-6)
		y3=np.cos(2*np.pi*1e3*t2)
		y3=xr.DataArray(	y3,
							dims=['t'],
							coords=[t2])
		y3.name='y3'
		
		num=int(np.random.rand()*10000)
		print(num)
		f = h5py.File('mytestfile%.5d.hdf5'%num, 'a')
		f = xr_DataArray_to_hdf5(y1, f ,path='data1')
		f = xr_DataArray_to_hdf5(y2, f, path='data2')
		f = xr_DataArray_to_hdf5(y3, f, path='data2')
		#f.close()
		
	Example 2::
		
		# 3D dataset with attributes
		x=np.arange(5)
		y=np.arange(10)
		z=np.arange(20)
		three_d_data=xr.DataArray(	np.random.rand(5,10,20),
								dims=['x','y','z'],
								coords=[x,y,z],
								attrs={	'name':'video1',
										'comment':'this is an example video',
										'long_name':'video 1 of data',
										'units':'au'})
		
		
		num=int(np.random.rand()*10000)
		print(num)
		f = h5py.File('mytestfile%.5d.hdf5'%num, 'a')
		f = xr_DataArray_to_hdf5(three_d_data, f, var_name='video', path='data1')

	"""

	# make sure the data has a name
	if var_name=='':
		if y.name==None:
			raise Exception('var_name not provided')
		else:
			var_name=y.name
			
	f = h5py_file
	
	# combine path and var_name and initialize the dataset
	var_name='%s/%s'%(path,var_name)
	f.create_dataset(var_name, data=y)
	
	# create and attach each dimension to the dataset
	for i,dim_name in enumerate(y.dims):
	
		f[var_name].dims[i].label = dim_name
		try:
			f['%s/'%path+dim_name]=y[dim_name].values
		except OSError:
			print('OSError: The time basis already existed for this group. Skipping...')  
			pass
		
		f['%s/'%path+dim_name].make_scale(dim_name)
		f[var_name].dims[i].attach_scale(f['%s/'%path+dim_name])
		
	# copy attributes
	for i,attr in enumerate(y.attrs):
		f[var_name].attrs.create(	attr,
								    y.attrs[attr])
		
	
	return f


def hdf5_to_xr_DataArray(	h5py_file, 
							var_path):
	"""
	
	Examples
	--------
	Example 1::
	
		# multiple time series data with different time bases
		t1=np.arange(0,1e-2,1e-6)
		y1=np.sin(2*np.pi*1e3*t1)
		y1=xr.DataArray(	y1,
							dims=['t'],
							coords=[t1])
		y1.name='y1'
		
		t2=np.arange(0,1e-2,2e-6)
		y2=np.cos(2*np.pi*1e3*t2)
		y2=xr.DataArray(	y2,
							dims=['t'],
							coords=[t2])
		y2.name='y2'
		# t2=np.arange(0,1e-2,2e-6)
		y3=np.cos(2*np.pi*1e3*t2)
		y3=xr.DataArray(	y3,
							dims=['t'],
							coords=[t2])
		y3.name='y3'
		
		num=int(np.random.rand()*10000)
		print(num)
		f = h5py.File('mytestfile%.5d.hdf5'%num, 'a')
		f = xr_DataArray_to_hdf5(y1, f ,path='data1')
		f = xr_DataArray_to_hdf5(y2, f, path='data2')
		f = xr_DataArray_to_hdf5(y3, f, path='data2')
		
		da1=hdf5_to_xr_DataArray(f,'data1/y1')
		da2=hdf5_to_xr_DataArray(f,'data2/y2')
		da3=hdf5_to_xr_DataArray(f,'data2/y3')
		
		f.close()
		
	Example 2::
		
		# 3D dataset with attributes
		x=np.arange(5)
		y=np.arange(10)
		z=np.arange(20)
		three_d_data=xr.DataArray(	np.random.rand(5,10,20),
								dims=['x','y','z'],
								coords=[x,y,z],
								attrs={	'name':'video1',
										'comment':'this is an example video',
										'long_name':'video 1 of data',
										'units':'au'})
		
		
		num=int(np.random.rand()*10000)
		print(num)
		f = h5py.File('mytestfile%.5d.hdf5'%num, 'a')
		f = xr_DataArray_to_hdf5(three_d_data, f, var_name='video', path='data1')

		da=hdf5_to_xr_DataArray(f,'data1/video')
		
		f.close()
	
	"""
			
	f = h5py_file
	
	data=f[var_path]#[()] 	# note that this allows the attrs to be coppied over
	
	# create dimensions and coordinates
	dims=[]
	coords=[]
	for a in f[var_path].dims: # presently, this only grabs the first dim from each dimension
		#print(a)
		dims.append(list(a)[0])
		coords.append(a[0][()])
		
	da=_xr.DataArray(	data,
						dims=dims,
						coords=coords)
	
	return da
