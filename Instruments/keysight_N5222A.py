# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:21:15 2021

@author: jwbrooks
"""

import johnspythonlibrary2 as jpl2
import nrl_code as nrl
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

import pyvisa
address='TCPIP0::A-N5221A-11119::inst0::INSTR'

class keysight_N5222A:
	# programming manual : http://na.support.keysight.com/pna/help/PNAHelp9_90.pdf
	
	def __init__(self, address='TCPIP0::A-N5221A-11119::inst0::INSTR'):
		self.address=address
		self.init_connection()
		
	def write_command(self,cmd, check_for_error=True):
		self.unit.write(cmd)
		if check_for_error == True:
			error=self.check_error()
			if "No error" not in error:
				print(error)
	# 		else:
	# 			print(error)
	
	def query(self, cmd, check_for_error=True):
		
		response=self.unit.query(cmd)
		if check_for_error == True:
			error=self.check_error()
			if "No error" not in error:
				print(error)
		return response	
				
			
	def fast_cw_stop(self):
		self.write_command('SENSe1:SWEep:TYPE:FACW 0')
		self.write_command("SYST:FIFO OFF")
		
	def fast_cw_start(self):
		# http://na.support.keysight.com/vna/help/latest/Programming/GPIB_Example_Programs/Setup_FastCW_and_FIFO.htm
		self.write_command("SENS:BWID:RES 600khz") # set IF Bandwidth to large value
		self.write_command("SENS:AVER:MODE POINT")

		self.write_command("SENS:AVER ON")

		self.write_command("SENS:AVER:COUNT 1")
		## good example on page 3192 of pdf
		# possibly a relevant example of page 3054 of pdf
		# 1 - Activechannel.hold 1 # hold syncronous
		self.write_command("SENS:SWE:MODE HOLD")
		# 2 - FIFO.State = 1 # turn on
		self.write_command("SYST:FIFO ON")
		# 3 - FIFO.Clear # clears buffer
		self.write_command("SYST:FIFO:DATA:CLEAR")
		# 4 - Turn on CW sweep type
		self.write_command("SENS:SWE:TYPE CW")
		# 5 - enable fast cw by specifying a the count size
		self.write_command('SENSe1:SWEep:TYPE:FACW 1000000') # set 0 to disable fast CW.  This is the number of points to measure in fast cw (before presumably it turns itself off)
		# 6 - Activechannel.single 1 # syncronous single
		self.write_command("SENS:SWE:MODE SINGle") # pg 3119 in programming manual

	def fast_cw_status(self):
		FCW_ON=bool(self.query("SENSe1:SWEep:TYPE:FACW?"))
# 		FCW_ON=bool(self.query('SYST:FIFO?'))
		print('Fast CW is On: ', FCW_ON)
		FCW_points_available=int(self.query("SYST:FIFO:DATA:COUNT?"))
		print('Points available: ', FCW_points_available)
		
	def fast_cw_get_data(self, count=None):
		if count is None:
			count=int(self.query("SYST:FIFO:DATA:COUNT?"))
		data=np.array(self.query("SYST:FIFO:DATA? %d"%count).split(',')).astype(float)
		data_real = data[::2]
		data_imag = data[1::2]
		data=data_real+1j*data_imag
			
		return data			 

	def convert_Z_to_S(self,Z, Z_ref=50+1j*0):
		S=(Z-Z_ref)/(Z+Z_ref)
		return S
		
	def convert_S_to_Z(self,S, Z_ref=50.0+1j*0):
		Z = Z_ref * ((1+S)/(1-S))
		return Z
		
	def init_connection(self): #TODO print a connection confirmation
		""" initialize connection with unit """
		rm = pyvisa.ResourceManager()
		self.unit = rm.open_resource(self.address)
# 		unit.timeout = self.timeout
		self.unit.clear()
		print("Connected to '%s' at '%s'"%(self.unit.query("*IDN?"), self.address))
		
	def set_sweep(self, startFreq=10e6, stopFreq=1e9, nPts=991, dwell=None):
		
		
		self.write_command('SENSe1:SWEep:TYPE LIN')
		if startFreq is not None:
			self.write_command(':SENS:FREQ:STAR %d' % np.max([int(1e7),int(startFreq)]))
		if stopFreq is not None:
			self.write_command(':SENS:FREQ:STOP %d' % int(stopFreq))
		if nPts is not None:
			self.write_command('SENS:SWE:POIN %d'%int(nPts))
		if dwell is not None:
			 self.write_command('SENS:SWE:DWEL %.12f'%float(dwell))
		 
		
	def set_avg(self, avg_on='ON', avg_count=10, if_bandwidth=30e3):
		self.unit.write(':SENS:AVER ' + avg_on + '\n')
		self.unit.write(':SENS:AVER:COUNT %d\n' % int(avg_count))
		self.unit.write(':SENS:BAND:RES %d\n' % int(if_bandwidth))

# 	def set_conversion(self, conv_on='ON', conv_func='ZREF'):
# 		self.unit.write(':CALC:SEL:CONV:STAT %s\n'%conv_on)		# %conv_on)
# 		self.unit.write(':CALC:SEL:CONV:FUNC %s\n'%conv_func)	# %conv_func)

	def set_format(self, fmat='MLIN'):  
		# MLIN = linear mag
		# MLOG = log mag
		# SCOM = Smith complex 
		# POL = Polar complex
		# http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/?nid=-11143.0.00&cc=US&lc=eng&id=1790874
		# http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/programming/command_reference/calculate/scpi_calculate_ch_selected_format.htm
		self.unit.write(':CALC:FORM %s'%fmat)  # %fmat)

	def check_error(self, verbose=False):
		error = self.unit.query("SYST:ERR?")
		if verbose:
			print(error)
		return error
	
	def setup_channel(self):
		self.unit.write("CALC:PAR:SEL 'CH1_S11_1'") # a lot of commands won't work unless the channel is setup like this
		
	def get_data(self, convert_Z=True):
		#TODO add a pause button to the data ?
		self.setup_channel()
		f=self.get_frequency()
		data=np.array(self.unit.query("CALC:DATA? SDATA").split(',')).astype(float)
		
		if len(f)*2 == len(data): # checks to see if the data is complex (i.e. contains both real and imag data and is therefore twice as long as the frequency)
			data_real = data[::2]
			data_imag = data[1::2]
			data=data_real+1j*data_imag
			
		data=self.convert_S_to_Z(data)
		return xr.DataArray(data,dims='f',coords=[f])
		
	def get_frequency(self):
		fstart=float(self.unit.query(':SENS:FREQ:STAR?' ))
		fstop=float(self.unit.query(':SENS:FREQ:STOP?' ))
		npoints=int(self.unit.query(':SENS:SWE:POIN?' ))
		
		freq = np.linspace(fstart,fstop,npoints)
		
		return freq
		
# 	def get_frequency_data(self):
# 		"""
# 		Get the list of frequencies of the instrument sweep, returning a
# 		sequence of floating point numbers.
# 		"""
# 		self.unit.query(":SENS1:FREQ:DATA?")
# 		frequency_data = b""
# 		while (frequency_data[len(frequency_data) - 1:] != b"\n"):
# 			frequency_data += self.vna_socket.recv(1024)
# 		frequency_data = frequency_data[:len(frequency_data) - 1].split(b",")
# 		frequency_data = [float(i) for i in frequency_data]
# 		return np.array(frequency_data)

		
if __name__ == "__main__":
	vna=keysight_N5222A()