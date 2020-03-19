
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt



class scpData:
	"""
	ssh and scp combination toolkit for file transfers
	
	Parameters
	----------
	password : str
		password for ssh server
	address : str
		address for ssh server
	port : int
		port number for ssh server
	username : str
		username for ssh server
 
	Attributes
	----------
	ssh : 'paramiko.SSHClient'
	scp : 'scp.SCPClient'
	
	Notes
	-----	
	
	References
	----------
	https://gist.github.com/stonefury/06ab3531a1c30c3b998a
	https://github.com/jbardin/scp.py
	
	Examples
	--------
	
	"""		
#	def __init__(self,password=getPwd(),username=_pref._HBT_SERVER_USERNAME,
#				 address=_pref._HBT_SERVER_ADDRESS,port=22):
					 
	def __init__(self,password,username,
				 address,port=22):
		from paramiko import SSHClient
		from scp import SCPClient
#		print('password=%s, username=%s' %(password,username))
		self.ssh = SSHClient()
		self.ssh.load_system_host_keys()
		self.ssh.connect(address, port=port, username=username, 
						 password=password)
		self.scp = SCPClient(self.ssh.get_transport())
		
	def downloadFile(self,remoteFilePath,localFilePath=''):
#		try:
		self.scp.get(remoteFilePath, localFilePath)
#		except:
#			print("%s not present.  skipping..." % remoteFilePath)
#			pass
		
	def uploadFile(self,localFilePath,remoteFilePath):
		self.scp.put(localFilePath,remoteFilePath)
		
	def uploadFolder(self,localDirPath,remoteDirPath):
		""" upload folder and all contents """
		self.scp.put(localDirPath,remoteDirPath,recursive=True)
		
	def closeConnection(self):
		self.scp.close();
		self.ssh.close();
		
	def __del__(self):
		""" upon deletion of object, close ssh/scp connections """
		self.closeConnection();

