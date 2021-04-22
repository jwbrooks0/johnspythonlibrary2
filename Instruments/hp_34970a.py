# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:57:14 2021

@author: jwbrooks
"""

# import johnspythonlibrary2 as jpl2
# import nrl_code as nrl
import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa
import socket
from time import sleep, time
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
	
	# https://documentation.help/Keysight-34970A-34972A/documentation.pdf
	
	def __init__(self, ip='192.168.0.243', gpib_address=b'9'):
		self.ip = ip
		self.gpib_address = gpib_address
		self.connect()
		sleep(0.1)
		self.init_gpib_eth()
		sleep(0.1)
		self.init_hp34970a()
		sleep(1)
		
	def disconnect(self):
		self.sock.close()
		
	def send_command(self,command, wait_time=0.1):
		self.sock.send( (command + '\n').encode() )
		sleep(wait_time)
		
	def receive_response(self,num_bytes=100000, max_wait_time=10):
		response=b''
		total_time=0
		wait_interval=0.5
		while(response[-1:] != b'\n' and total_time<max_wait_time):
# 			print(total_time, response)
# 			if total_time>max_wait_time:
# 				break
			if total_time>0 and response==b'':
				break
			sleep(wait_interval)
			total_time+=wait_interval
# 			print(total_time, wait_time)
			try:
				response+=self.sock.recv(num_bytes)
			except:
				pass
# 			print(response)

		#print('done', response)
			
		return response
	
	def send_and_receive(self,command,wait_time=0.1,num_bytes=100000):
		self.send_command(command, wait_time=wait_time)
		return self.receive_response(num_bytes)
		
	def connect(self):
		# if having trouble connecting, try the GPIB Configurator app from the prologix website
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
		sock.settimeout(1)
		sock.connect((self.ip, 1234))
		
		# confirm connection by printing connection details
		sock.send(b"++ver\n")
		sleep(1)
		print(b"Connected to '%s'"%sock.recv(100000))
		print(sock)
		
		self.sock = sock
		
	def init_gpib_eth(self):
		# Set mode as CONTROLLER
		self.sock.send(b"++mode 1\n")
		
		# Set HP33120A address
		self.sock.send(b"++addr %s\n"%self.gpib_address)
		
	def init_hp34970a(self, mode='T', channels='@201:208'):
		#TODO I need a confirmation command that I'm actually talking with this unit.
		#TODO the commands in this section need to be vetted.
		if True:
			self.sock.send(b'*RST\n')	# A Factory Reset (*RST command) turns off the units, time, channel, and alarm information
			self.sock.send(b':ABORt\n')	# stops a scan
	# 		self.sock.send(b':CONFigure:TEMPerature %s,%s,(%s)\n' % (b'TCouple', b'K', b'@102:104'))
	# 		self.sock.send(b':CONFigure:VOLT:DC 1,(@101:120') # configures 1V range at channels 101 to 120
			if mode =='V':
				self.sock.send(b':CONFigure:VOLT:DC AUTO,(%s)\n'%channels.encode()) # configures auto range at channels 101 to 120
			elif mode=='T':
				self.sock.send(b':CONFigure:TEMPerature %s,%s,(%s)\n' % (b'TCouple', b'K', channels.encode()))
				
# 			self.sock.send(b':CONFigure:VOLT:DC AUTO,(@101:122') # configures auto range at channels 101 to 120
	# 		self.sock.send(b':UNIT:TEMPerature %s\n' % (b'C'))
			self.sock.send(b':ROUTe:SCAN (%s)\n' % (channels.encode())) # configures channels 101 to 122 to be scanned
			self.sock.send(b':TRIGger:SOURce %s\n' % (b'TIMer')) # setup a timer to automatically trigger the scan
			self.sock.send(b':TRIGger:COUNt %d\n' % (1)) #number of scans
			self.sock.send(b':TRIGger:TIMer %G\n' % (1.0)) # trigger interval in seconds
			self.sock.send(b':FORMat:READing:CHANnel %d\n' % (1)) # 1 includes channel number in returned data
			self.sock.send(b':FORMat:READing:ALARm %d\n' % (0)) # 0 removes alarm inforation from returned data
			self.sock.send(b':FORMat:READing:UNIT %d\n' % (1)) # 1 includes the unit for measured data
			self.sock.send(b':FORMat:READing:TIME:TYPE %s\n' % (b'ABSolute')) # sets the time format type
			self.sock.send(b':FORMat:READing:TIME %d\n' % (1)) # 1 includes the time in the returned data
	# 		self.sock.send(b':SYSTem:TIME %.2d,%.2d,%s\n' % (1)) # print('%.2d.%s'%(a,('%.3f'%a)[2:]))
		
		if False: # This is backup that works.  Saving here in case of sanity check			
			self.sock.send(b'*RST\n')	# A Factory Reset (*RST command) turns off the units, time, channel, and alarm information
			self.sock.send(b':ABORt\n')	# stops a scan
			self.sock.send(b':CONFigure:TEMPerature %s,%s,(%s)\n' % (b'TCouple', b'K', b'@102:104'))
	# 		self.sock.send(b':CONFigure:VOLT:DC 1,(@101:120') # configures 1V range at channels 101 to 120
# 			self.sock.send(b':CONFigure:VOLT:DC AUTO,(@101:122') # configures auto range at channels 101 to 120
			self.sock.send(b':UNIT:TEMPerature %s\n' % (b'C'))
			self.sock.send(b':ROUTe:SCAN (%s)\n' % (b'@101:122')) # configures channels 101 to 122 to be scanned
			self.sock.send(b':TRIGger:SOURce %s\n' % (b'TIMer')) # setup a timer to automatically trigger the scan
			self.sock.send(b':TRIGger:COUNt %d\n' % (1)) #number of scans
			self.sock.send(b':TRIGger:TIMer %G\n' % (1.0)) # trigger interval in seconds
			self.sock.send(b':FORMat:READing:CHANnel %d\n' % (1)) # 1 includes channel number in returned data
			self.sock.send(b':FORMat:READing:ALARm %d\n' % (0)) # 0 removes alarm inforation from returned data
			self.sock.send(b':FORMat:READing:UNIT %d\n' % (1)) # 1 includes the unit for measured data
			self.sock.send(b':FORMat:READing:TIME:TYPE %s\n' % (b'ABSolute')) # sets the time format type
			self.sock.send(b':FORMat:READing:TIME %d\n' % (1)) # 1 includes the time in the returned data
	# 		self.sock.send(b':SYSTem:TIME %.2d,%.2d,%s\n' % (1)) # print('%.2d.%s'%(a,('%.3f'%a)[2:]))
		
			
		
	def _get_data_raw(self, wait_time=10):
		self.init_hp34970a()
		self.receive_response() #clear buffer
		sleep(0.1)
		self.sock.send(b':READ?\n')
		return self.receive_response(max_wait_time=wait_time)

		
	def get_data(self, wait_time=10, plot=False):
		
		raw_data=self._get_data_raw(wait_time=wait_time)
		raw_data=np.array(raw_data.decode('ascii').split(',')).reshape(-1,8)

		data=np.zeros(raw_data.shape[0])
		channels=np.zeros(raw_data.shape[0],dtype=int)
		for i in range(len(data)):
			data[i]=np.float(raw_data[i,0].split(' ')[0])
			channels[i]=int(raw_data[i,7])
		
		data_out = xr.DataArray( data,
							  dims='ch',
							  coords=[channels])
		
		if plot==True:
			fig,ax=plt.subplots()
			data_out.plot(ax=ax)
		
		return data_out
	
	def slow_aquire_data(self, wait_time_in_min=1, filename='out.csv', headers=['Chamber','Anode','TDU','Breadboard','Anode_cooling_block','Z-stage','R-stage','PIP_mount'], mode='T', channels='@201:208'):#, plot=False):
		self.init_hp34970a(mode=mode,channels=channels)
		
		import os
		from datetime import datetime
		from johnspythonlibrary2.ReadWrite import append_single_row_to_csv
		now=datetime.now().strftime("%H:%M:%S")
		#now.strftime("%H:%M:%S")
		data=self.get_data()
		if os.path.exists(filename):
			headers=[]
		else:
			headers=['time', *headers]
		append_single_row_to_csv( [now, *data.data.astype(str)], filename=filename, headers=headers)
# 		if plot==True:
# 			self.plot_time_results_animate()
		while(True):
			sleep(int(wait_time_in_min*60))
			now=datetime.now().strftime("%H:%M:%S")
			data=self.get_data()
			append_single_row_to_csv( [now, *data.data.astype(str)], filename=filename)
	
		# TODO write function that plots the results and autoupdates the plot each cycle
	
	def plot_time_results(filename='out.csv'):
		import pandas as pd
		data=pd.read_csv(filename).set_index('time')
		
			
		
	
if __name__ == '__main__':
	unit = hp_34970a_prologix('192.168.0.105')
	print(unit.send_and_receive('++ver'))
# # 	print(unit.receive_response())
# 	raw_data=unit._get_data_raw(50)
# 	print(raw_data)
# # 	print(unit._get_data_raw(1))
# 	fig,ax=plt.subplots()
# 	ch=1
# 	data=unit.get_data()
# 	data.plot(ax=ax)
# 	ax.legend()
	unit.slow_aquire_data(1, 'test30_temperature.csv')
# 	unit.disconnect()
# 	