
import numpy as _np
import os as _os
import importlib.util as _util
from pathlib import Path as _Path
# from collections import Sequence as _Sequence
from numpy.distutils.misc_util import is_sequence as _is_sequence


def meshgrid_to_NDarray(arrs):
    """
    I often find that I need meshgrid to generate points that are in an NxM array form (N number of M-dimensional points) instead of M number of N**M matrices.
    This code does this.
    
    Examples
    --------
    
    Example 1::
        
        import numpy as np
        
        ## test tupple of numpy arrays
        x = np.random.rand(10,1)
        y = np.random.rand(10,1)
        z = np.random.rand(10,1)
        arrs = (x, y, z)
        meshed_points = meshgrid_to_NDarray(arrs)
        print("passed")
        
        ## test list of numpy arrays
        x = np.random.rand(10,1)
        y = np.random.rand(10,1)
        z = np.random.rand(10,1)
        arrs = [x, y, z]
        meshed_points = meshgrid_to_NDarray(arrs)
        print("passed")
        
        ## test if arrays can be both horizontal and vertical
        x = np.random.rand(10)
        y = np.random.rand(10)
        z = np.random.rand(10)
        arrs = (x, y, z)
        meshed_points = meshgrid_to_NDarray(arrs)
        print("passed")
        
        ## test if arrays can have different lengths; should fail
        x = np.random.rand(10)
        y = np.random.rand(11)
        z = np.random.rand(12)
        arrs = (x, y, z)
        meshed_points = meshgrid_to_NDarray(arrs)
        print("failed")
        
        
    Notes
    -----
     * Based loosely on code found here: https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points
        
    """
    
    # number of arrays
    dim = len(arrs)
    
    # length of each array
    lens = _np.array(list(map(len, arrs)))
    assert _np.all(lens == lens[0]), "all arrays must have the same length"
    
    
    out = _np.meshgrid(*arrs)
    result = _np.zeros((lens[0] ** dim, dim))
    for i in range(dim):
        result[:, i] = out[i].flatten()
    
    return result


def is_sequence(var):
	"""
	Returns True if var is a sequence (list, array, tuple, dicts, etc) or False otherwise.
	Note that strings are not sequences (even though they can be indexed and have a len(var) >= 0 )
	
	References
	----------
	 * https://stackoverflow.com/questions/2937114/python-check-if-an-object-is-a-sequence

	"""
	return _is_sequence(var)
	# import collections
	# return isinstance(var, collections.Sequence)


def import_module_from_file(file_path):
	""" 
	Imports a specific python file as a module/library 
	
	Parameters 
	----------
	file_path : str of pathlib.Path
		filepath to a .py file to import
		
	Returns
	-------
	module : module
		Module/library containing the file
		
	Example
	-------
	
		file_path = r'Z:\Programs\PIPPIn\PIP_experimental_data\2022_December_testing_campaign\process_pip_sweeps.py'
		pps = import_module_from_file(file_path)
	"""
	module_name = _Path(file_path).stem
	spec = _util.spec_from_file_location(module_name, file_path)
	module = _util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


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


