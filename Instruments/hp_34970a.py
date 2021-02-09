# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:57:14 2021

@author: jwbrooks
"""

# import johnspythonlibrary2 as jpl2
# import nrl_code as nrl
import numpy as np
# import matplotlib.pyplot as plt
import pyvisa as visa
import socket
import time
import xarray as xr

## Working progress

## references
# https://community.keysight.com/thread/20818
# https://community.keysight.com/thread/22576
# https://community.keysight.com/thread/27047


def under_construction():
	# this code actually works but needs to be cleaned up
	rm = visa.ResourceManager()
	v34972A = rm.open_resource('GPIB0::9::INSTR')
	v34972A.timeout = None
	v34972A.write('*RST')
	v34972A.write(':ABORt')
	v34972A.write(':CONFigure:TEMPerature %s,%s,(%s)' % ('TCouple', 'K', '@102:104'))
	v34972A.write(':UNIT:TEMPerature %s' % ('C'))
	v34972A.write(':ROUTe:SCAN (%s)' % ('@101:122'))
	v34972A.write(':TRIGger:SOURce %s' % ('TIMer'))
	v34972A.write(':TRIGger:COUNt %d' % (1)) #number of scans
	v34972A.write(':TRIGger:TIMer %G' % (1.0))
	v34972A.write(':FORMat:READing:CHANnel %d' % (1))
	v34972A.write(':FORMat:READing:ALARm %d' % (0))
	v34972A.write(':FORMat:READing:UNIT %d' % (1))
	v34972A.write(':FORMat:READing:TIME:TYPE %s' % ('ABSolute'))
	v34972A.write(':FORMat:READing:TIME %d' % (1))
	# v34972A.write(':FORMat:READing:UNIT %d' % (1))
	readings = v34972A.query(':READ?')
	v34972A.close()
	rm.close()
	
	
class hp_34970a_prologix:
	
	def __init__(self, ip='192.168.0.105', gpib_address=b'9'):
		self.ip = ip
		self.gpib_address = gpib_address
		self.connect()
		self.init_gpib_eth()
		self.init_hp34970a()
		
	def disconnect(self):
		self.sock.close()
		
	def connect(self):

		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
		sock.settimeout(0.1)
		sock.connect((self.ip, 1234))
		
		# confirm connection by printing connection details
		sock.send(b"++ver\n")
		time.sleep(0.1)
		print(b"Connected to '%s'"%sock.recv(100000))
		print(sock)
		
		self.sock = sock
		
	def init_gpib_eth(self):
		# Set mode as CONTROLLER
		self.sock.send(b"++mode 1\n")
		
		# Set HP33120A address
		self.sock.send(b"++addr %s\n"%self.gpib_address)
		
	def init_hp34970a(self):
		
		#TODO the commands in this section need to be vetted.
		self.sock.send(b'*RST\n')
		self.sock.send(b':ABORt\n')
		self.sock.send(b':CONFigure:TEMPerature %s,%s,(%s)\n' % (b'TCouple', b'K', b'@102:104'))
		self.sock.send(b':UNIT:TEMPerature %s\n' % (b'C'))
		self.sock.send(b':ROUTe:SCAN (%s)\n' % (b'@101:122'))
		self.sock.send(b':TRIGger:SOURce %s\n' % (b'TIMer'))
		self.sock.send(b':TRIGger:COUNt %d\n' % (1)) #number of scans
		self.sock.send(b':TRIGger:TIMer %G\n' % (1.0))
		self.sock.send(b':FORMat:READing:CHANnel %d\n' % (1))
		self.sock.send(b':FORMat:READing:ALARm %d\n' % (0))
		self.sock.send(b':FORMat:READing:UNIT %d\n' % (1))
		self.sock.send(b':FORMat:READing:TIME:TYPE %s\n' % (b'ABSolute'))
		self.sock.send(b':FORMat:READing:TIME %d\n' % (1))
		
	def get_data(self, wait_time=5):
		
		self.sock.send(b':READ?\n')
		time.sleep(wait_time)
		raw_data=np.array(self.sock.recv(100000).decode('ascii').split(',')).reshape(-1,8)

		data=np.zeros(raw_data.shape[0])
		channels=np.zeros(raw_data.shape[0],dtype=int)
		for i in range(len(data)):
			data[i]=np.float(raw_data[i,0].split(' ')[0])
			channels[i]=int(raw_data[i,7])
		
		data_out = xr.DataArray( data,
							  dims='ch',
							  coords=[channels])
		
		return data_out
	
	