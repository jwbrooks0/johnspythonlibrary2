# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:48:28 2021

@author: jwbrooks
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time
import pyvisa as visa

class keithley_24xx:
	
	#TODO add a function that sets the voltage range
	
	def __init__(self,address='TCPIP0::192.168.0.236::inst0::INSTR',timeout=5000):
		
		self.address=address
		self.timeout=timeout
		self.connect()
		
	def connect(self):
		rm=visa.ResourceManager()
		print(rm.list_resources())
		self.instrument=rm.open_resource(self.address)
		print('Connected to : ' + self.instrument.query("*IDN?"))
		
		self.instrument.timeout=self.timeout
		
		self.instrument.write("sour:func volt")
		
	def disconnect(self):
		
		self.instrument.close()
		
	def set_voltage(self,v):
		
		self.instrument.write("sour:volt %d"%v)
		
	def output_on(self):
		
		self.instrument.write(':OUTPut:STATe ON')
		self.instrument.write(':OUTP:STAT ON')
		
	def output_off(self):
		
		self.instrument.write(':OUTP OFF')
		
	def read_values(self,count=1):
		results=np.zeros(count)
		for i in range(count):
			results[i]= np.array(self.instrument.query_ascii_values(':READ?'))
		return results

	def set_current_limit(self,i_limit):
		
		self.instrument.write("sour:volt:ilimit %.3f"%i_limit)

	def IV_sweep(self,v_range,i_max=1,count=5,settling_time=0.1,plot=False):
		
		results=np.zeros(len(v_range))
		
		for i,v in enumerate(v_range):
			self.set_voltage(v)
			time.sleep(settling_time)
			current=self.read_values(count=count).mean()
			print(v,current)
			results[i]=current
		da=xr.DataArray(	results,
							  dims='V',
							  coords=[v_range])
		
		if plot==True:
			fig,ax=plt.subplots()
			da.plot(ax=ax)
		
		return da
	
if __name__=='__main__':

	k24xx=keithley_24xx()
	k24xx.set_voltage(10)
	k24xx.set_current_limit(.1)
	k24xx.output_on()
	data=k24xx.read_values(10)
	IV_data=k24xx.IV_sweep(np.arange(-10,11),plot=True)
	k24xx.output_off()
	k24xx.disconnect()

# import usb

## file:///C:/Users/jwbrooks/Downloads/2450-900-01_D_May_2015_User_3.pdf

# k2450.write("*rst")
# k2450.write("sour:func volt")
# k2450.write("sour:volt:rang 20")
# k2450.write("sour:volt:ilimit 0.02")

# inst=k2450

# k2450.write(':ROUT: REAR')
# print(inst.query(':ROUT?'))
# inst.write(':SENS:FUNC:CONC OFF')
# inst.write(':SOUR:FUNC CURR')
# inst.write(':SENS:FUNC VOLT:DC')
# inst.write(':SENS:VOLT:PROT 10')
# inst.write(':SOUR:CURR:START 1E-3')
# inst.write(':SOUR:CURR:STOP 10E-3')
# inst.write(':SOUR:CURR:STEP 1E-3')
# inst.write(':SOUR:CURR:MODE SWE')
# inst.write(':SOUR:SWE:RANG AUTO')
# inst.write(':SOUR:SWE:SPAC LIN')
# inst.write(':TRIG:COUN 10')
# inst.write(':SOUR:DEL 0.1')

# k2450.close()
# 	 
	 