"""
Functions to help optomize code
"""

import numpy as _np

def functionTimer(func):
	"""
	Decorator for timing a function.  Performs the calculations 'iter' times.  
	Prints mean and std time.  Returns an array of times.
	
	Example
	-------
	Example1 ::
		
		@functionTimer
		def waste_some_time(num_times,a=1):
			for _ in range(num_times):
				sum([i**2 for i in range(10000)])
		
		a=waste_some_time(10,iters=50)
		
	References
	----------
	* Code borrowed from here with some tweaks - https://realpython.com/lessons/timing-functions-decorators/
	"""
	
	from engineering_notation import EngNumber # converts a number to engineering format with units!
	import functools
	import time
	
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		
		# Number of iterations.  Looks for a user provided number.  Else, uses a default.
		try:
			iters=kwargs.pop('iters')
		except:
			iters=100
		
		# Run code 'iter' times.  
		result=_np.zeros(iters)
		for i in range(iters):
			start_time = time.perf_counter()  # 1
			value = func(*args, **kwargs)
			end_time = time.perf_counter()  # 2
			run_time = end_time - start_time  # 3
			result[i]=run_time
			
		# print and return result
		print(f"Finished {func.__name__!r} in {str(EngNumber(result.mean()))}s +- {str(EngNumber(result.std()))}s in {iters:d} interations")
		return result
	return wrapper_timer


