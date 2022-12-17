
import numpy as _np
import os as _os


def get_immediate_subdirectories(a_dir):
	"""
	Returns all subdirectores within a_dir
	
	 * https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python

	Parameters
	----------
	a_dir : str
		directory

	Returns
	-------
	list
		list of subdirectories

	"""
	return [name for name in _os.listdir(a_dir)
		 if _os.path.isdir(_os.path.join(a_dir, name))]


def time_as_str(format_str="%Y_%m_%d_%H_%M_%S"):
	from datetime import datetime
	
	return datetime.now().strftime(format_str) 


def str_to_time(date_time, format_str="%Y_%m_%d_%H_%M_%S"):
	from datetime import datetime
	
	return datetime.strptime(date_time, format_str)


def binary_to_int(bin_num):
	""" 
	Converts binary number to int 
	
	Parameters
	----------
	bin_num : str
		Binary number.  E.g. bin(100) = '0b1100100'
		
	Returns
	-------
	int
		Integer (converted from binary)
		
	"""
	return int(bin_num, 2)


def int_to_binary(int_num):
	""" Converts int to binary """
	return bin(int_num)


def round_float_to_arbitrary_resolution(value, resolution):
	"""
	
	Examples
	--------
	Examples ::
		
		round_float_to_resolution(10.021, 0.025)
		round_float_to_resolution(3.141592653589793, 0.001)

	"""
	return round(value / resolution) * resolution


def createLibrarySimple(keys, vals):
	"""
	Simple method to create library by passing a list of keys and a list of 
	values (vals)
	
	Parameters
	----------
	keys : list of strings
		keys for the library
	vals : list
		values(vals) to be associated with each key
		
	Example
	-------
	::
		
		keys = ['a', 'b', 'c']
		vals = [1, 2, 3]
		zipped = createLibrarySimple(keys,vals)
		print(zipped)
	"""
	if len(keys) != len(vals):
		raise Exception('Lengths of keys and vals are not equal')
	return dict(zip(keys, vals))


def retrieveVariableName(var):
	"""
	Returns the "name" of a variable as a string
	
	Parameters
	----------
	var : variable or list of variables
		single variable or list of variables
	
	Returns
	-------
	name : string
		name of the variable 
		
	Examples
	--------
	::
		
		# int
		a=1
		retrieveVariableName(a)
		
		# numpy array
		import numpy as np
		asdf=np.arange(0,10)
		retrieveVariableName(asdf)
		
		# list of variables
		a=1
		b=2
		c=3
		retrieveVariableName([a,b,c])
	"""
	import inspect as _inspect
	
	callers_local_vars = _inspect.currentframe().f_back.f_locals.items()
	if type(var) != list:
		name = [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
	else:
		print(_np.shape(var))
		name = []
		for n in var:
			name.append([var_name for var_name, var_val in callers_local_vars if var_val is n][0])
			
	return name


