
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt



def setPwd(password,system,username):
	""" 
	Encrypts password using keyring, a password management tool.  
	
	Use in conjunction with getPwd()
	
	Parameters
	----------
	password : str
		ssh password for hbtep server
	system : str
		name of ssh hbtep server.  
	username : str
		name of login on ssh hbtep server.  
	
	NOTES
	-----
	this function also ONLY works on a work station where the OS-based 
	function keyring is installed and the password has already been set for 
	that user.  i'm using this on ubuntu.  not sure if it'll work on windows
	"""
	import keyring
	keyring.set_password(system,username,password)
	
	
def getPwd(system,username):
	""" 
	Returns unencrypted password
	
	Parameters
	----------
	password : str
		ssh password for hbtep server
	system : str
		name of ssh hbtep server.  
	username : str
		name of login on ssh hbtep server.  
		
	Returns
	-------
	: str
		ssh password 
	
	NOTES
	-----
	this function also ONLY works on a work station where the OS-based 
	function keyring is installed and the password has already been set for 
	that user.  i'm using this on ubuntu.  not sure if it'll work on windows
	"""
	import keyring
	return str(keyring.get_password(system, username))




def playBeep(durationInS=0.1,freqInHz=440):
	"""
	Play short beep
	
	References
	----------
	https://stackoverflow.com/questions/16573051/sound-alarm-when-code-finishes
	"""
	import os
	os.system('play -nq -t alsa synth {} sine {}'.format(durationInS, freqInHz))


def checkAndCreateDir(directory):
	"""
	Checks to see if a directory exists and creates it if not
	"""
	import os
	if not os.path.exists(directory):
		os.makedirs(directory)
		

def todaysDate(formatting="standard"):
	"""
	References
	----------
	https://www.programiz.com/python-programming/datetime/current-datetime
	"""
	from datetime import date
	today=date.today()
	if formatting=="standard":
		return today.strftime("%d/%m/%Y")
	elif formatting=="underscore":
		return today.strftime("%Y_%m_%d")


def listDirectoriesForAllModules():
	import sys
	a=sys.path
	for i in range(0,len(a)):
		print("%s" % a[i])
		
	return a


def returnPythonVersion():
	import sys
	return sys.version_info.major



def deprecated(func):
	"""
	This is a decorator which can be used to mark functions as deprecated. 
	It will result in a warning being emitted when the function is used.
	
	References
	----------
	https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically/30253848#30253848
	
	Examples
	--------
	::
		
		@deprecated
		def some_old_function(x, y):
			return x + y
			
		some_old_function(1,2)
			
	::
		
		class SomeClass:
			@deprecated
			def some_old_method(self, x, y):
				return x + y
		a=SomeClass()
		a.some_old_method(2,3)
	"""
	import warnings
	import functools
	
	@functools.wraps(func)
	def new_func(*args, **kwargs):
		warnings.simplefilter('always', DeprecationWarning)  # turn off filter
		warnings.warn("Call to deprecated function {}.".format(func.__name__),
					  category=DeprecationWarning,
					  stacklevel=2)
		warnings.simplefilter('default', DeprecationWarning)  # reset filter
		return func(*args, **kwargs)
	return new_func



