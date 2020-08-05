
# import numpy as _np
# import pandas as _pd
# import matplotlib.pyplot as _plt

import os as _os


def processFileName(filename):
	"""
	Extracts teh base name, directory name, file extension, and file root from
	a filename.

	Parameters
	----------
	filename : str
		filename with directory information.

	Returns
	-------
	baseName : str
		The basename of the file (filename without directory)
	dirName : str
		Directory name where the file is located
	fileExt : str
		File extension
	fileRoot : str
		Filename without the extension

	Examples
	--------
	Example 1::
		
		filename='/media/john/T7-Blue/asdf/awesome_data.csv'
		out=processFileName(filename)
		print(out)
		baseName,dirName,fileExt,fileRoot=out
		
	Example 2::
		
		print(processFileName('asdf.123'))
		
	Example 3::
		
		print(processFileName('asdf/123'))
	"""
	fileRoot,fileExt=_os.path.splitext(filename)
	baseName=_os.path.basename(filename)
	dirName=_os.path.dirname(filename)
	
	return baseName,dirName,fileExt,fileRoot


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
	try:
		os.system('play -nq -t alsa synth {} sine {}'.format(durationInS, freqInHz))
	except:
		print('This command only works on linux. \n If play not found error provided, you need to install play.  In ubuntu, type: sudo apt install sox')


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






