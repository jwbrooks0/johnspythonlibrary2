

##############################################################################
# %% Import libraries
import numpy as _np
import pandas as _pd
# import matplotlib.pyplot as _plt
from functools import wraps as _wraps # allows doc string to be visible through the decorator
import pickle as _pkl
import xarray as _xr
from pathlib import Path as _Path
# from glob import glob as _glob
import os as _os
import shutil as _shutil # download data
from datetime import datetime as _datetime

try:
	import johnspythonlibrary2.ReadWrite._settings as _settings
	LOCALDIRTOSAVEDATA = _settings.localDirToSaveData
except ImportError:
	print('Warning: ReadWrite/_settings.py file not found.  Have you modified the ReadWrite/_settingsTemplate.py file yet?')
	LOCALDIRTOSAVEDATA = ''
	
	

##############################################################################
# %% timestamps for filenames
def datetime_now_to_str(fmt="%Y_%m_%d_%H_%M_%S"): # fmt="%Y_%d_%m_%H_%M_%S_%f"
    
    """
    
    # Examples
    # --------
    
    # Example 1::
        
    #     from datetime import datetime as _datetime
    #     time = _datetime(2024, 5, 9, 10, 0, 0, 0)
    #     time = _datetime.now().strftime(fmt)
    #     print(time)
    #     fmt="%Y_%m_%d_%H_%M_%S"
    #     time.strftime(fmt)
        
    """
    return _datetime.now().strftime(fmt)


def str_to_datetime(
        date_string, 
        fmt="%Y_%m_%d_%H_%M_%S",
        # fmt="%Y_%m_%d_%H_%M_%S.%f",
        ):
    """
    
    Examples
    --------
    
    Example 1::
        
        date_string = "2024_05_09_07_47_12"
        fmt = "%Y_%m_%d_%H_%M_%S"
        time = str_to_datetime(date_string, fmt)
        print(time)
        
    Example 2::
        
        date_string = ["2024_05_09_07_47_12",
                       "2024_05_09_07_47_13",
                       "2024_05_09_07_47_14", ]
        fmt = "%Y_%m_%d_%H_%M_%S"
        times = str_to_datetime(date_string, fmt)
        print(times)
        
        
    """
    if type(date_string) is str:
        return _datetime.strptime(date_string, fmt) 
    elif len(date_string) > 0:
        results = []
        for i, ds in enumerate(date_string):
            results.append(str_to_datetime(ds, fmt))
        return results




##############################################################################
# %% manage local vs remote data

def store_file_locally(remote_filepath, 
					   local_dir = r'C:\Users\spacelab\python\local_data_copy', 
					   verbose=True,
					   force_update=False):
	""" This function checks to see if the file exists on the local computer.  If not there, it downloads it there for the future.  """
	
	## convert all paths to pathlib
	local_dir = _Path(local_dir)
	remote_filepath = _Path(remote_filepath)
	
	## construct local filepath
	local_filepath = local_dir / remote_filepath.name
	
	## check if file is local
	if _os.path.exists(local_filepath) is False or force_update is True: # if not local, download a copy
		if verbose: print('File not found locally.  Downloading... ', end='')
		_shutil.copyfile(remote_filepath, local_filepath)
		if verbose: print('Done. ')
	else:
		if verbose: print('File found locally. ')
	
	return local_filepath
	

##############################################################################
# %% file I/O misc

def delete_file_if_exists(filepath):
	import os
	if os.path.exists(filepath) is True:
	    os.remove(filepath)
	
	
##############################################################################
# %% excel

def excel_to_pandas(filepath, 
					sheet_name=0, # 0 returns the first sheet, 1 is the second, "Sheet1" returns that sheet, etc
					header_row=None, # specify the header row.  integer or None expected
					header_names=None, # specify header names.  List of strings or None.
					index_col=None, # column of index.  integer or None expected
					):
	
	return _pd.read_excel(filepath, 
					sheet_name=sheet_name,
					header=header_row,
					names=header_names,
					index_col=index_col)



##############################################################################
# %% csv

def xr_dataarray_to_csv(data, filename='test.csv'):
	""" writes xarray dataarray to csv file """
	out = data.to_pandas()
	out.name = data.name
	out.to_csv(filename)
	

def csv_to_xr(filename, 
              delimiter=',', 
              row_number_of_col_names='infer', 
              first_column_is_index=True, 
              number_of_rows=None, 
              dim_dtype=None, 
              skiprows=None,
              dtype=None,
              ):

	# filename='C:\\Users\\jwbrooks\\python\\nrl_code\\vna_impedance\\test29_mikeN_balun_board_S_measurements\\sn3_cal1.csv'
	
	data = _pd.read_csv(filename, delimiter=delimiter, header=row_number_of_col_names, skip_blank_lines=True, skiprows=skiprows)
	
	if type(number_of_rows) != type(None):
		data = data.iloc[:number_of_rows, :]
		
	if first_column_is_index is True:
		data = data.set_index(data.iloc[:, 0].name)
		
	if type(dim_dtype) != type(None):
		data.index = data.index.astype(dim_dtype)
		
	data = data.to_xarray()
	
	keys = list(data.keys())
	
	return data[keys[0]]



def csv_to_pd(filename, delimiter=',', row_number_of_col_names='infer', first_column_is_index=True, number_of_rows=None, dim_dtype=None, skiprows=None):

	# filename='C:\\Users\\jwbrooks\\python\\nrl_code\\vna_impedance\\test29_mikeN_balun_board_S_measurements\\sn3_cal1.csv'
	
	data = _pd.read_csv(filename, delimiter=delimiter, header=row_number_of_col_names, skip_blank_lines=True, skiprows=skiprows)
	
	if type(number_of_rows) != type(None):
		data = data.iloc[:number_of_rows, :]
		
	if first_column_is_index is True:
		data = data.set_index(data.iloc[:, 0].name)
		
	if type(dim_dtype) != type(None):
		data.index = data.index.astype(dim_dtype)
	
	return data


def append_single_row_to_csv(data_row, filename='name.csv', headers=[], delete_file_if_already_exists=False): 
	"""
	Examples
	--------
	Example 1 ::
		
		import numpy as np
		headers=['a','b','c']
		filename='filename_csv.csv'
		for i in np.arange(10):
			data=np.random.rand(len(headers))
			if i==0:
				append_single_row_to_csv(data, filename=filename,headers=headers)
			else:
				append_single_row_to_csv(data, filename=filename)

	"""
	import csv   
	
	if delete_file_if_already_exists is True:
		delete_file_if_exists(filename)

	with open(filename, 'a', newline='') as f:
		writer = csv.writer(f)
		if len(headers) > 0:
			writer.writerow(headers)
		writer.writerow(data_row)
		
		
#################################################
# %% Pickles

def pickle_to_any_data(filename):
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
	
	return _pkl.load(open(filename, 'rb'))


def any_data_to_pickle(data, filename):
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
	
	_pkl.dump(data, open(filename, "wb"))
	

##############################################################################
# %% Matlab data

def matlab73_to_xr_DataArray(filename, dataNames=None, returnRaw=False):
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
	data_temp = mat73.loadmat(filename)
	
	if returnRaw is True:
		return data_temp
	else:
		colNames = _np.array(list(data_temp.keys()), dtype=str)[:-1]
		if type(dataNames) == type(None):
			dataNames = colNames.copy()
			
		das = {}
		for ic, c in enumerate(colNames):
			t = _np.arange(data_temp[c].NumPoints) * data_temp[c].XInc + data_temp[c].XOrg
			da = _xr.DataArray((data_temp[c].Data) * data_temp[c].YInc + data_temp[c].YOrg,
								dims=['t'],
								coords={'t': t},
								attrs={'units': data_temp[c].YUnits})
			
			da.t.attrs = {'units': data_temp[c].XUnits}
			das[dataNames[ic]] = da
			
		return _xr.Dataset(das)
	
	
def mat_to_dict(filename):
	
	# libraries
	from scipy.io import loadmat

	# load .mat file into a dictionary
	matData = loadmat(filename)
	
	# get the column (header) names for each array
	names = list(matData.keys())
	
	print('loaded:', names)
	
	return matData
	

# def mat_to_pd_DataFrame(filename, keys=[]):
# 	"""
# 	Converts a matlab .mat file to a pandas dataframe
# 	
# 	Parameters
# 	----------
# 	filename : str
# 		path and filename of the file to be processed
# 		
# 	Returns 
# 	-------
# 	df : pandas dataframe
# 		dataframe containing the same data and headers as the .mat file
# 		
# 	"""
# 	
# 	# load .mat file into a dictionary
# 	matData=mat_to_dict(filename)
# 	
# # 	get the column (header) names for each array
# # 	names=list(matData.keys())
# # 	colNames=[]
# # 	for i in range(len(names)):
# # 		colNames.append(names[i])
# 	
# 	# convert the dict to a dataframe with the appropriate column names
# 	df=_pd.DataFrame()
# 	for i in keys:
# 		print(i)
# 		df[i]=matData[i]
# 	
# 	return df


#################################################
# %% HBT related
def backupDFs(func, defaultDir=LOCALDIRTOSAVEDATA, debug=False):
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
			
		name = args[0]
		if debug:
			print(name)
			
		if 'forceDownload' in kwargs:
			forceDownload = kwargs.pop('forceDownload')
		else:
			forceDownload = False
		if debug:
			print(forceDownload)
			
		partialFileName = '%s_%s' % (str(name), func.__name__)
		if debug:
			print(partialFileName)
		
		try:
			if LOCALDIRTOSAVEDATA == '':
				raise Exception('Default directory not specified.  Forcing download.')
			elif forceDownload is True:
				raise Exception('forcing download')
			import glob
			inList = glob.glob('%s%s*.pkl' % (defaultDir, partialFileName))
			if len(inList) == 0:
				print('data does not exist, downloading')
				raise Exception('data does not exist, downloading')
			elif len(inList) == 1:
				fileName = '%s%s.pkl ' % (defaultDir, partialFileName)
				df = _pd.read_pickle(fileName)
				print('loaded %s from memory' % fileName)
			else:
				df = []
				for i in range(len(inList)):
					fileName = '%s%s_%d.pkl' % (defaultDir, partialFileName, i)
					df.append(_pd.read_pickle(fileName))
					print('loaded %s from memory' % fileName)
		except:
			read = func(*args, **kwargs)
			if type(read) == _pd.core.frame.DataFrame:
				if LOCALDIRTOSAVEDATA != '':
					fileName = '%s%s_%s.pkl' % (defaultDir, str(name), func.__name__)
					read.to_pickle(fileName)
					print('downloading and storing %s' % fileName)
			else:
				for i in range(len(read)):
					df = read[i]
					if LOCALDIRTOSAVEDATA != '':
						fileName = '%s%s_%s_%d.pkl' % (defaultDir, str(name), func.__name__, i)
						df.to_pickle(fileName)
						print('downloading and storing %s' % fileName)
			df = read
			
		return df
					
	return inner1


##############################################################################
# %% ODS files

def ods_to_pd_DataFrame(filename, sheetName='Sheet1', header=0):
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
	df = _pd.DataFrame({col[header].value:[x.value for x in col[header + 1:]]
						for col in tab.columns()})
	
	return df


##############################################################################
# %% Videos

def np_array_to_mp4(array, file_name):
	"""
	
	Parameters
	----------
	array : np.ndarray
		Video data.  Dimensions = (t, y, x, color)
	file_name : str
		Name of mp4 file to write the data to
		
	Examples
	--------
	Example 1:: # black and white
		
		import numpy as np
		array = np.random.random(size=(50, 480, 680, 1)) * 255
		array = array.astype(np.uint8)
		file_name="outputvideo_BW.mp4"
		np_array_to_mp4(array,file_name=file_name)
		
	Example 2:: # color
		
		import numpy as np
		array = np.random.random(size=(50, 480, 680, 3)) * 255
		array = array.astype(np.uint8)
		file_name="outputvideo_color.mp4"
		np_array_to_mp4(array,file_name=file_name)

	References
	----------
	*  http://www.scikit-video.org/stable/io.html

	"""
	
	import skvideo.io as _skio
	
	print("work in progress")
	if not file_name.endswith('.mp4'):
		file_name += '.mp4'
		
	_skio.vwrite(file_name, array)


##############################################################################
# %% Text files, generic

def str_to_text_file(input_str, filename):
	""" 
	Writes a str of data to a single row in a text file. 
	This is useful, for instance, if a csv file is being written to one line at a time. 
	"""
	text_file = open(filename, "w")
	text_file.write(input_str)	
	text_file.close()


##############################################################################
# %% HDF5

# ## proposed hdf5 data structure template for future data
# Tier 1 (Parent directory) : Experiment # or case
# Tier 2 : Parameter scan(s)
# Tier 3 : Instruments/Diagnostics (because datasets with different time bases should not be mixed)
# Tier 4 : Raw and Processed data

def get_hdf5_tree(hdf5_file_path, text_file_output_name=None):
	""" Returns entire hdf5 file tree to the screen """
	import nexusformat.nexus as nx
	f = nx.nxload(hdf5_file_path)
	out = f.tree
	f.close()
	
	if type(text_file_output_name) == str:
		str_to_text_file(out, text_file_output_name)
		print('wrote data to : %s' % text_file_output_name)
	
	return out


def hdf5_add_metadata_old(	hdf5_item, library):
	""" Add a library as attributes to an hdf5 item (group or dataset) """
	
	for key in library.keys():
		hdf5_item.attrs.create(name=key, data=library[key])
		
		
def hdf5_add_metadata(	hdf5_filename, library):
	""" Add a library as attributes to an hdf5 item (group or dataset) """
	print('not working yet')
	import h5py
	f = h5py.File(hdf5_filename, mode='a')
	for key in library.keys():
		f.attrs.create(name=key, data=library[key])
		
		
# def xr_Dataset_to_hdf5(		ds,
# 							hdf5_file_name,
# 							group_name):
# 	"""
# 	The xarray library has a convenient method of converting a dataset to an hdf5 file.  This is a wrapper for this.
# 	
# 	Parameters
# 	----------
# 	ds : xarray.core.dataset.Dataset
# 		Dataset to save
# 	hdf5_file_name : str
# 		Filename of the hdf5 to be created or appended to
# 	group_name : str
# 		internal hdf5 path to save the data
# 	"""
# 	ds.to_netcdf(hdf5_file_name, 
# 					mode='a', 
# 					format='NETCDF4', 
# 					group=group_name, 
# 					engine='h5netcdf', 
# 					invalid_netcdf=True)
	
	
def xr_Dataset_to_hdf5(	ds,
							hdf5_file_name,
							group_name="/",
							compression_level=2):
	"""
	The xarray library has a convenient method of converting a dataset to an hdf5 file.  This is a wrapper for this.
	
	Parameters
	----------
	ds : xarray.core.dataset.Dataset
		Dataset to save
	hdf5_file_name : str
		Filename of the hdf5 to be created or appended to
	group_name : str
		internal hdf5 path to save the data
		
	References
	----------
	* https://stackoverflow.com/questions/40766037/specify-encoding-compression-for-many-variables-in-xarray-dataset-when-write-to
	"""
	# add encoding (compression) for each variable in the dataset
	comp = dict(compression='gzip', compression_opts=compression_level)
	encoding = {var: comp for var in ds.data_vars}
	
	# write to hdf5 file
	ds.to_netcdf(hdf5_file_name, 
					mode='a', 
					format='NETCDF4', 
					group=group_name, 
					engine='h5netcdf', 
					invalid_netcdf=True,
					encoding=encoding)


def xr_DataArray_to_hdf5(	da, hdf5_file_name, group_name="/", compression_level=5):
	""" Writes xarray DataArray to hdf5 file format """
	
	if da.name is None:
		da.name = 'data'
	
	xr_Dataset_to_hdf5(		ds=da.to_dataset(),
							hdf5_file_name=hdf5_file_name,
							group_name=group_name,
							compression_level=compression_level)


def hdf5_to_xr_Dataset(		hdf5_file, group_name="/"):
	ds = _xr.load_dataset(hdf5_file, group=group_name, engine="h5netcdf")
	return ds
	

def hdf5_to_xr(		hdf5_file, group_name="/"):
	""" 
	Reads hdf5 data and returns xarray dataset if multiple variables or xarray dataarray is a single variable 
	
	Parameters 
	----------
	hdf5_file : str
		name and path of file
	group_name : str
		group name of data (or dataset name)
	"""
	ds = hdf5_to_xr_Dataset(hdf5_file=hdf5_file, group_name=group_name)
	if len(ds) == 1:
		return ds[list(ds.data_vars)[0]]
	else:
		return ds


def xr_DataTree_to_hdf5(dt, hdf5_filename, mode='w'):
	
	## write data to file
	dt.to_netcdf(	hdf5_filename, 
					mode=mode,
					# format='NETCDF4', # I think that NETCDF4 is the default.
					engine='h5netcdf',
					# encoding=encoding, # TODO implement encoding to allow for compression
					invalid_netcdf=True
					)
	
	
def hdf5_to_xr_DataTree():
	print("presently not implemented.  I think there is a bug in the DataTree code that needs to be resolved first.")
	
	# example code
	if False:
		import numpy as np
		import xarray as xr
		import datatree as dt
	
	
		## create example datatree
		t1 = np.arange(0, 1e0, 1e-4)
		t1 = xr.DataArray(t1, dims='t', coords=[t1], attrs={'units': 's', 'long_name': 'time'})
		t2 = np.arange(0, 1e-1, 1e-6)
		t2 = xr.DataArray(t2, dims='t', coords=[t2], attrs={'units': 's', 'long_name': 'time'})
	
		a1 = xr.DataArray(np.random.rand(len(t1)), dims='t', coords=[t1], name='data_A1', attrs={'units': 'au', 'long_name': 'data A1'})
		a2 = xr.DataArray(np.random.rand(len(t1)), dims='t', coords=[t1], name='data_A2', attrs={'units': 'au', 'long_name': 'data A2'})
		a = xr.Dataset({a1.name: a1, a2.name: a2})
	
		b1 = xr.DataArray(np.random.rand(len(t2)), dims='t', coords=[t2], name='data_B', attrs={'units': 'au', 'long_name': 'data B'})
		b = b1.to_dataset()
	
		c1 = xr.DataArray(np.random.rand(len(t2)), dims='t', coords=[t2], name='data_C1', attrs={'units': 'au', 'long_name': 'data C1'})
		c2 = xr.DataArray(np.random.rand(len(t2)), dims='t', coords=[t2], name='data_C2', attrs={'units': 'au', 'long_name': 'data C2'})
		c = xr.Dataset({c1.name: c1, c2.name: c2})
	
		data = dt.DataTree.from_dict({'data1': a, 'data2/b': b, 'data2/c': c})
	
		## write data to file
		data.to_netcdf(	'example_data_1.hdf5', 
						mode='w',
		# 				format='NETCDF4', 
						engine='h5netcdf',
						# encoding=encoding,
						invalid_netcdf=True
						)
	
		## read data from file
		data2 = dt.open_datatree('example_data_1.hdf5', 
								  engine='h5netcdf')
		
		return data2
