# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:26:17 2021

@author: jwbrooks
"""

# import johnspythonlibrary2 as jpl2
# import nrl_code as nrl
# import numpy as np
# import matplotlib.pyplot as plt
import pyvisa
# import sys

			
class keysight_n5700:
	""" 
	connects to, controls, and downloads data from they Keysight N5710A DC power supply 
	
	TODO: This function has not been checked (at all)
	
	Notes
	-----
	*  Manual : https://literature.cdn.keysight.com/litweb/pdf/5969-2917.pdf
		
	"""
		
	debug=False
	
	def __init__(	self,
					address="TCPIP0::192.168.0.226::inst0::INSTR",
					timeout=20000,
					debug=False):
		self.address=address
		self.timeout=timeout
		self.connect()
		self.debug=debug
		
	def disconnect(self):
		self.instrument.close()
		print("Disconnected from '%s' at '%s'" % (self.idn_string, self.address))
		
	def connect(self):
		""" initialize connection with unit """
		rm = pyvisa.ResourceManager()
		instrument = rm.open_resource(self.address)
		instrument.timeout = self.timeout
		instrument.clear()
		self.instrument=instrument
		self.idn_string = self.do_query_string("*IDN?")
		print("Connected to '%s' at '%s'" % (self.idn_string, self.address))
		
	def set_voltage(self, voltage):
		self.do_command(command="VOLT %.3f" % voltage)
		
	def turn_on_overcurrent_protection(self, on=True):
		if on==True:
			self.do_command("CURR:PROT:STAT 1")
		else:
			self.do_command("CURR:PROT:STAT 0")			
		
	def set_max_current(self, i_max):
		self.do_command(command="CURR %.3f" % i_max)
		
	def set_output_on(self, on=True):
		if on == True:
			self.do_command(command="OUTP ON")
		else:
			self.do_command(command="OUTP OFF")
			
	def measure_volt_and_current(self):
		
		v=self.do_query_string("Meas:Volt?")
		i=self.do_query_string("Meas:Curr?")
		
		return v,i
		
	def do_command(self, command):
		if self.debug:
			print("\nCmd = '%s'" % command)
		self.instrument.write("%s" % command)
		self.check_instrument_errors(command)
		
	def check_instrument_errors(self, command):
		
		error_string = self.instrument.query("Syst:err?")
		if "No error" not in error_string:
			raise Exception("Error '%s' encountered from command '%s'"%(error_string, command))
			
				
	def do_query_string(self,query):
		""" Send a query, check for errors, return string: """
		if self.debug:
			print("Qys = '%s'" % query)
		result = self.instrument.query("%s" % query)
		self.check_instrument_errors(query)
		return result