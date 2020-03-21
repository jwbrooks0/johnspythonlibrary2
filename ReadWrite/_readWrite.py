
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from functools import wraps as _wraps


try:
	import johnspythonlibrary2.ReadWrite._settings as _settings
except ImportError:
	raise Exception('Code hault: settings.py file not found.')
	



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


def backupDFs(func,defaultDir=_settings.localDirToSaveData,debug=False):
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
			if forceDownload==True:
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
				fileName='%s%s_%s.pkl'%(defaultDir,str(name),func.__name__)
				read.to_pickle(fileName)
				print('downloading and storing %s'%fileName)
			else:
				for i in range(len(read)):
					df=read[i]
					fileName='%s%s_%s_%d.pkl'%(defaultDir,str(name),func.__name__,i)
					df.to_pickle(fileName)
					print('downloading and storing %s'%fileName)
			df=read
			
		return df
					
	return inner1




#@backupDFs
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
