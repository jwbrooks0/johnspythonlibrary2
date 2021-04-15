# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 09:47:18 2021

@author: jwbrooks
"""


import time
import numpy as np
import xarray as xr
# import matplotlib.pyplot as plt
import socket

class agilent_E5061B:
	"""
	Class used to send commands and acquire data from the Agilent E5061B vector \
		network analyzer.
		
	* Manual incl programming : https://grandline.jahschwa.com/~kiosk/equip/agilent/e5061b/agilent-e5061b-manual.pdf
	Note: Make sure you enable telnet communication.  System -> Misc Setup -> \
		Network Setup -> Enable Telnet
	http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/?nid=-32496.1150148.00&cc=US&lc=eng&id=1790874

	Note: This code is based on the code here: https://github.com/lnls-dig/instr
		_tests/blob/master/instr_tests/instruments/vna/agilent_e5061b.py
	"""

	def __init__(self, ip="192.168.0.111", port=5025, sleep_time=0.5):
		"""
		Class constructor.
		Here the socket connection to the instrument is initialized. The
		argument required, a string, is the IP adress of the instrument.
		"""
		# open connection
		self.vna_address = ((ip, port))
		self.SLEEP_TIME = sleep_time
		self.open_connection()
		
		
	def initial_setup(self):
		# set default settings
		self.set_avg()
		self.set_sweep()
		self.set_conversion()
		self.set_format()
		self.auto_scale()

		# self.vna_socket.send(b":DISP:WIND1:TRAC1:Y:RLEV -40\n")
		# self.vna_socket.send(b":DISP:WIND1:TRAC1:Y:PDIV 15\n") # scale per devision (does not work in smith chart)
		# self.vna_socket.send(b":DISP:WIND{1-4}:TRAC{1-4}:Y:AUTO\n")
		# self.vna_socket.send(b":SENS1:SWE:TIME:AUTO ON\n")
		# self.vna_socket.send(b":SENS1:SWE:POIN 1601\n")
		# self.vna_socket.send(b":SENS1:SWE:TYPE LIN\n")
		# self.vna_socket.send(b":SOUR1:POW:GPP 0.0\n")

	def open_connection(self):
		self.vna_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.vna_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		self.vna_socket.settimeout(5.0)
		self.vna_socket.connect(self.vna_address)
		#self.vna_socket.send(b":SYST:PRES\n")
		print('successfully connected at %s' % str(self.vna_address))
		time.sleep(self.SLEEP_TIME)
		
		
	def set_single_freq(self, freq):
		"""set the span of the VNA"""
		#TODO change number of points to 2
		#TODO set avg number to 1
		self.vna_socket.send(b":SENS1:FREQ:CENT " + str(freq).encode()+b'\n')
		self.vna_socket.send(b":SENS1:FREQ:SPAN " + str(0).encode()+b'\n')
		self.vna_socket.send(b":SENS1:SWE:TIME:AUTO OFF\n" );
		self.vna_socket.send(b":SENS1:SWE:TIME %.3f\n" % 1);
		
		time.sleep(self.SLEEP_TIME)


	def get_answer(self, max_num_of_bytes=1024):
		"""
		Get the instrument's answer after sending a command.
		It is returned as a string of bytes.
		"""
		data = b""
		while (data[len(data) - 1:] != b"\n"):
			data += self.vna_socket.recv(max_num_of_bytes)
		return(data)

	def get_frequency_data(self):
		"""
		Get the list of frequencies of the instrument sweep, returning a
		sequence of floating point numbers.
		"""
		self.vna_socket.send(b":SENS1:FREQ:DATA?\n")
		frequency_data = b""
		while (frequency_data[len(frequency_data) - 1:] != b"\n"):
			frequency_data += self.vna_socket.recv(1024)
		frequency_data = frequency_data[:len(frequency_data) - 1].split(b",")
		frequency_data = [float(i) for i in frequency_data]
		return np.array(frequency_data)

	
	def _get_answer_complex(self, max_num_of_bytes=1024):
		"""
		Get the instrument's answer after sending a command.
		It is returned as a string of bytes.
		"""
		data=self.get_answer(max_num_of_bytes=max_num_of_bytes)
		data = data[:len(data) - 1].split(b",")
		data_real = data[::2]
		data_imag = data[1::2]
# 		data_real = [round(float(i), 5) for i in data_real]
# 		data_imag = [round(float(i), 5) for i in data_imag]
		data_real = [float(i) for i in data_real]
		data_imag = [float(i) for i in data_imag]
		data_complex = np.array(data_real) + 1j * np.array(data_imag)
			
		return(data_complex)
	
	
	def get_data(self, types=['s11','s12','s21','s22']):
		
		data=[]
		freq = self.get_frequency_data()
		if 's11' in types or 'all' == types:
			self.send_command(b":CALC1:PAR1:DEF S11\n")
			time.sleep(self.SLEEP_TIME)
			self.send_command(':CALC1:PAR1:SEL')
			self.send_command(b":CALC1:DATA:FDAT?\n")
			s11_data=self._get_answer_complex()
			s11_data=xr.DataArray( s11_data, dims='f', coords=[freq], attrs={'name':'s11'})
			s11_data.name='s11'
			data.append(s11_data)
		if 's12' in types or 'all' == types:
			self.send_command(b":CALC1:PAR2:DEF S12\n")
			time.sleep(self.SLEEP_TIME)
			self.send_command(':CALC1:PAR2:SEL')
			self.send_command(b":CALC1:DATA:FDAT?\n")
			s12_data=self._get_answer_complex()
			s12_data=xr.DataArray( s12_data, dims='f', coords=[freq], attrs={'name':'s12'})
			s12_data.name='s12'
			data.append(s12_data)
		if 's21' in types or 'all' == types:
			self.send_command(b":CALC1:PAR3:DEF S21\n")
			time.sleep(self.SLEEP_TIME)
			self.send_command(':CALC1:PAR3:SEL')
			self.send_command(b":CALC1:DATA:FDAT?\n")
			s21_data=self._get_answer_complex()
			s21_data=xr.DataArray( s21_data, dims='f', coords=[freq], attrs={'name':'s21'})
			s21_data.name='s21'
			data.append(s21_data)
		if 's22' in types or 'all' == types:
			self.send_command(b":CALC1:PAR4:DEF S22\n")
			time.sleep(self.SLEEP_TIME)
			self.send_command(':CALC1:PAR4:SEL')
			self.send_command(b":CALC1:DATA:FDAT?\n")
			s22_data=self._get_answer_complex()
			s22_data=xr.DataArray( s22_data, dims='f', coords=[freq], attrs={'name':'s22'})
			s22_data.name='s22'
			data.append(s22_data)
			
		return xr.merge(data)


	def send_command(self, command):
		"""
		Sends a command to the instrument. 
		The "text" argument must be a string of bytes.
		"""
		if type(command) == str:  # make sure the command is binary
			command=command.encode()
		if command[-1:]!='\n'.encode(): # make sure the command ends in "\n"
			command+='\n'.encode()
		self.vna_socket.send(command)
		time.sleep(self.SLEEP_TIME)
		return


	def close_connection(self):
		"""Close the socket connection to the instrument."""
		self.vna_socket.close()
		print("connection closed")


	def auto_scale(self):
		self.vna_socket.send(b':DISP:WIND1:TRAC1:Y:AUTO\n')


	def set_avg(self, avg_on='ON', avg_count=10, if_bandwidth=30e3):
		self.vna_socket.send(b':SENS1:AVER ' + bytearray(avg_on, 'utf-8') + b'\n')
		self.vna_socket.send(b':SENS1:AVER:COUNT %d\n' % avg_count)
		self.vna_socket.send(b':SENS1:BAND:RES %d\n' % if_bandwidth)


	def set_sweep(self, f_start=1e6, f_stop=1e9, n_points=1000, power=10): #default power = 0.  
		self.vna_socket.send(b':SENS1:FREQ:STAR %d\n' % f_start)
		self.vna_socket.send(b':SENS1:FREQ:STOP %d\n' % f_stop)
		self.vna_socket.send(b':SENS1:SWE:POIN %d\n' % n_points)
		self.vna_socket.send(b':SOUR1:POW %d\n' % power)
		self.vna_socket.send(b":SENS1:SWE:TIME:AUTO ON\n" );


	def set_conversion(self, conv_on='ON', conv_func='ZREF'):
		self.vna_socket.send(b':CALC1:SEL:CONV:STAT %s\n'%conv_on.encode())		# %conv_on)
		self.vna_socket.send(b':CALC1:SEL:CONV:FUNC %s\n'%conv_func.encode())	# %conv_func)


	def set_format(self, fmat='SCOM'):  
		# MLOG = maglog 
		# SCOM=Smith complex 
		# POL = Polar complex
		# http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/?nid=-11143.0.00&cc=US&lc=eng&id=1790874
		# http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/programming/command_reference/calculate/scpi_calculate_ch_selected_format.htm
		self.vna_socket.send(b':CALC1:FORM POL\n')  # %fmat)

