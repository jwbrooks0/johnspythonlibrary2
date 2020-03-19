
import numpy as _np
import inspect as _inspect


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
	callers_local_vars = _inspect.currentframe().f_back.f_locals.items()
	if type(var)!=list:
		name = [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
	else:
		print(_np.shape(var))
		name=[]
		for n in var:
			name.append([var_name for var_name, var_val in callers_local_vars if var_val is n][0])
			
	return name


