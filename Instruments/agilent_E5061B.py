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

	Note: Make sure you enable telnet communication.  System -> Misc Setup -> \
		Network Setup -> Enable Telnet
	http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/?nid=-32496.1150148.00&cc=US&lc=eng&id=1790874

	Note: This code is based on the code here: https://github.com/lnls-dig/instr
		_tests/blob/master/instr_tests/instruments/vna/agilent_e5061b.py
	"""

	def __init__(self, ip="192.168.0.111", port=5025, sleep_time=2.0):
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

	def get_answer(self):
		"""
		Get the instrument's answer after sending a command.
		It is returned as a string of bytes.
		"""
		data = b""
		while (data[len(data) - 1:] != b"\n"):
			data += self.vna_socket.recv(1024)
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

	def get_s11_data(self):
		"""Get the S11 trace data, returning a sequence of floating point numbers."""
		self.vna_socket.send(b":CALC1:PAR1:DEF S11\n")
		time.sleep(self.SLEEP_TIME)
		self.vna_socket.send(b":CALC1:DATA:FDAT?\n")
		s11_data = b""
		while (s11_data[len(s11_data) - 1:] != b"\n"):
			s11_data += self.vna_socket.recv(1024)
		s11_data = s11_data[:len(s11_data) - 1].split(b",")
		s11_data = s11_data[::2]
		s11_data = [round(float(i), 2) for i in s11_data]
		return(s11_data)

	def get_s11_data_complex(self):
		"""Get the S11 trace data, returning a sequence of floating point numbers."""
		self.vna_socket.send(b":CALC1:PAR1:DEF S11\n")
		time.sleep(self.SLEEP_TIME)
		self.vna_socket.send(b":CALC1:DATA:FDAT?\n")
		s11_data = b""
		while (s11_data[len(s11_data) - 1:] != b"\n"):
			s11_data += self.vna_socket.recv(1024)
		s11_data = s11_data[:len(s11_data) - 1].split(b",")
		s11_data_real = s11_data[::2]
		s11_data_imag = s11_data[1::2]
		s11_data_real = [round(float(i), 2) for i in s11_data_real]
		s11_data_imag = [round(float(i), 2) for i in s11_data_imag]
		s11_data_complex = np.array(s11_data_real) + 1j * np.array(s11_data_imag)
		return s11_data_complex

	def send_command(self, text):
		"""
		Sends a command to the instrument. 
		The "text" argument must be a string of bytes.
		"""
		self.vna_socket.send(text)
		time.sleep(self.SLEEP_TIME)
		return

#  	def set_power(self, power):
# 		self.vna_socket.send(b":SOUR1:POW:GPP " + str(power) + b"\n")
# 		return

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

	def set_sweep(self, f_start=10e6, f_stop=1e9, n_points=1601, power=0):
		self.vna_socket.send(b':SENS1:FREQ:STAR %d\n' % f_start)
		self.vna_socket.send(b':SENS1:FREQ:STOP %d\n' % f_stop)
		self.vna_socket.send(b':SENS1:SWE:POIN %d\n' % n_points)
		self.vna_socket.send(b':SOUR1:POW %d\n' % power)

	def set_conversion(self, conv_on='ON', conv_func='ZREF'):
		self.vna_socket.send(b':CALC1:SEL:CONV:STAT ON\n')		# %conv_on)
		self.vna_socket.send(b':CALC1:SEL:CONV:FUNC ZREF\n')	# %conv_func)

	def set_format(self, fmat='SCOM'):  
		# MLOG = maglog 
		# SCOM=Smith complex 
		# http://ena.support.keysight.com/e5061b/manuals/webhelp/eng/?nid=-11143.0.00&cc=US&lc=eng&id=1790874
		self.vna_socket.send(b':CALC1:FORM SCOM\n')  # %fmat)

	def get_data(self, data_type='s11'):
		if data_type == 's11':
			data = self.get_s11_data_complex()
		else:
			raise Exception('Improper type requested: %s' % str(data_type))
		freq = self.get_frequency_data()
		da = xr.DataArray(data, dims=['f'], coords=[freq])
		da.f.attrs['units'] = 'Hz'
		da.f.attrs['long_name'] = 'Frequency'
		da.attrs['units'] = 'Ohm'
		da.name = 'Amplitude'

		return da
