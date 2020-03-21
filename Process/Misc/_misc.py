
import numpy as _np
import pandas as _pd
#import matplotlib.pyplot as _plt



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
	numbers=re.findall(r'\d+',string)
	
	# convert to integers
	for i in range(0,len(numbers)):
		numbers[i]=int(numbers[i])
		
	return numbers
	 

def findNearest(array,value):
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
	if type(array)==_np.ndarray:
		index = (_np.abs(array-value)).argmin()
	elif type(array)==_pd.core.frame.DataFrame:
		index = (_np.abs(array-value)).idxmin().values
		
	# value = array[index] 
	return index 
	# return index, value # uncomment to return both the index AND the value
	
