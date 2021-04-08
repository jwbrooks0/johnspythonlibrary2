#!python3
# *********************************************************
# This program illustrates a few commonly-used programming
# features of your Keysight Infiniium Series oscilloscope.
# *********************************************************
# Import modules.
# ---------------------------------------------------------
# import visa
import pyvisa
# import string
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
# # Global variables (booleans: 0 = False, 1 = True).
# # ---------------------------------------------------------
# debug = 0
# # =========================================================
# # Initialize:
# # =========================================================
# def initialize():
# 	# Clear status.
# 	do_command("*CLS")
# 	# Get and display the device's *IDN? string.
# 	idn_string = do_query_string("*IDN?")
# 	print("Identification string: '%s'" % idn_string)

# 	# Load the default setup.
# 	do_command("*RST")
# 	
# # =========================================================
# # Capture:
# # =========================================================
# def capture():
# 	# Set probe attenuation factor.
# # 	do_command(":CHANnel1:PROBe 1.0")
# 	qresult = do_query_string(":CHANnel1:PROBe?")
# 	print("Channel 1 probe attenuation factor: %s" % qresult)
# 	# Use auto-scale to automatically set up oscilloscope.
# 	print("Autoscale.")
# 	do_command(":AUToscale")
# 	# Set trigger mode.
# 	do_command(":TRIGger:MODE EDGE")
# 	qresult = do_query_string(":TRIGger:MODE?")
# 	print("Trigger mode: %s" % qresult)
# 	# Set EDGE trigger parameters.
# 	do_command(":TRIGger:EDGE:SOURce CHANnel1")
# 	qresult = do_query_string(":TRIGger:EDGE:SOURce?")
# 	print("Trigger edge source: %s" % qresult)
# # 	do_command(":TRIGger:LEVel CHANnel1,330E-3")
# 	qresult = do_query_string(":TRIGger:LEVel? CHANnel1")
# 	print("Trigger level, channel 1: %s" % qresult)
# 	do_command(":TRIGger:EDGE:SLOPe POSitive")
# 	qresult = do_query_string(":TRIGger:EDGE:SLOPe?")
# 	print("Trigger edge slope: %s" % qresult)
# 	# Save oscilloscope setup.
# # 	sSetup = do_query_ieee_block(":SYSTem:SETup?")
# # 	f = open("setup.stp", "wb")
# # 	f.write(sSetup)
# # 	f.close()
# # 	print("Setup bytes saved: %d" % len(sSetup))
# 	# Change oscilloscope settings with individual commands:
# 	# Set vertical scale and offset.
# 	do_command(":CHANnel1:SCALe 0.1")
# 	qresult = do_query_number(":CHANnel1:SCALe?")
# 	print("Channel 1 vertical scale: %f" % qresult)
# 	do_command(":CHANnel1:OFFSet 0.0")
# 	qresult = do_query_number(":CHANnel1:OFFSet?")
# 	print("Channel 1 offset: %f" % qresult)
# 	# Set horizontal scale and offset.
# 	do_command(":TIMebase:SCALe 200e-6")
# 	
# 	qresult = do_query_string(":TIMebase:SCALe?")
# 	print("Timebase scale: %s" % qresult)
# 	do_command(":TIMebase:POSition 0.0")
# 	qresult = do_query_string(":TIMebase:POSition?")
# 	print("Timebase position: %s" % qresult)
# 	# Set the acquisition mode.
# 	do_command(":ACQuire:MODE RTIMe")
# 	qresult = do_query_string(":ACQuire:MODE?")
# 	print("Acquire mode: %s" % qresult)
# # 	# Or, set up oscilloscope by loading a previously saved setup.
# # 	sSetup = ""
# # 	f = open("setup.stp", "rb")
# # 	sSetup = f.read()
# # 	f.close()
# # 	do_command_ieee_block(":SYSTem:SETup", sSetup)
# # 	print("Setup bytes restored: %d" % len(sSetup))
# 	# Set the desired number of waveform points,
# 	# and capture an acquisition.
# 	do_command(":ACQuire:POINts 32000")
# 	do_command(":DIGitize")
# 	
# 	
# # def get_all_data():
# # 	
# # 	return get_data(1),get_data(2),get_data(3),get_data(4)

# # def get_data(ch,plot=True):
# # 	
# # 	print(do_query_string(":WAVeform:POINts?"))
# # 	do_command(":ACQuire:POINts 32000")
# # 	print(do_query_string(":WAVeform:POINts?"))
# # # 	do_command(":DIGitize")
# # 	
# # 	do_command(":WAVeform:SOURce CHANnel%s"%ch)
# # 	do_command(":MEASure:SOURce CHANnel%s"%ch)
# # # 	do_command(":WAVeform:SOURce CHANnel1")
# # 	
# # 	print(do_query_string(":WAVeform:POINts?"))
# # 	do_command(":WAVeform:FORMat BYTE")	
# # 	
# # 	print(do_query_string(":WAVeform:POINts?"))
# # # 	do_command(":MEASure:SOURce CHANnel1")
# # # 	qresult = do_query_string(":MEASure:SOURce?")
# # # 	print("Measure source: %s" % qresult)
# # 	
# # 	print( do_query_string(":WAVeform:SOURce?"))
# # 	print( do_query_string(":MEASure:SOURce?"))
# # 		# Get the waveform data.
# # 	do_command(":WAVeform:STReaming OFF")
# # 	sData = do_query_ieee_block(":WAVeform:DATA?",data_points=int(do_query_string(":WAVeform:POINts?")))
# # 	# Unpack signed byte data.
# # 	values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
# # 	# print("Number of data values: %d" % len(values))
# # 	
# # 	x_increment = do_query_number(":WAVeform:XINCrement?")
# # 	x_origin = do_query_number(":WAVeform:XORigin?")
# # 	y_increment = do_query_number(":WAVeform:YINCrement?")
# # 	y_origin = do_query_number(":WAVeform:YORigin?")
# # 	
# # 	data = np.array(values) * y_increment + y_origin
# # 	time = np.arange(len(data)) * x_increment + x_origin

# # 	import xarray as xr
# # 	data=xr.DataArray(	data,
# # 							 dims='t',
# # 							 coords=[time])
# # 	
# # 	if plot==True:
# # 		fig,ax=plt.subplots()
# # 		data.plot(ax=ax)
# # 	return data
# # =========================================================
# # Analyze:
# # =========================================================
# def analyze():
# 	# Make measurements.
# 	# --------------------------------------------------------
# 	do_command(":MEASure:SOURce CHANnel1")
# 	qresult = do_query_string(":MEASure:SOURce?")
# 	print("Measure source: %s" % qresult)
# 	do_command(":MEASure:FREQuency")
# 	qresult = do_query_string(":MEASure:FREQuency?")
# 	print("Measured frequency on channel 1: %s" % qresult)
# 	do_command(":MEASure:VAMPlitude")
# 	qresult = do_query_string(":MEASure:VAMPlitude?")
# 	print("Measured vertical amplitude on channel 1: %s" % qresult)
# # 	# Download the screen image.
# # 	# --------------------------------------------------------
# # 	sDisplay = do_query_ieee_block(":DISPlay:DATA? PNG")
# # 	# Save display data values to file.
# # 	f = open("screen_image.png", "wb")
# # 	f.write(sDisplay)
# # 	f.close()
# # 	print("Screen image written to screen_image.png.")
# 	# Download waveform data.
# 	# --------------------------------------------------------
# 	
# 	# Get the waveform type.
# 	qresult = do_query_string(":WAVeform:TYPE?")
# 	print("Waveform type: %s" % qresult)
# 	# Get the number of waveform points.
# 	qresult = do_query_string(":WAVeform:POINts?")
# 	print("Waveform points: %s" % qresult)
# 	# Set the waveform source.
# 	do_command(":WAVeform:SOURce CHANnel1")
# 	qresult = do_query_string(":WAVeform:SOURce?")
# 	print("Waveform source: %s" % qresult)
# 	# Choose the format of the data returned:
# 	do_command(":WAVeform:FORMat BYTE")
# 	print("Waveform format: %s" % do_query_string(":WAVeform:FORMat?"))
# 	# Display the waveform settings from preamble:
# 	wav_form_dict = {
# 	0 : "ASCii",
# 	1 : "BYTE",
# 	2 : "WORD",
# 	3 : "LONG",
# 	4 : "LONGLONG",
# 	}
# 	acq_type_dict = {
# 	1 : "RAW",
# 	2 : "AVERage",
# 	3 : "VHIStogram",
# 	4 : "HHIStogram",
# 	6 : "INTerpolate",
# 	10 : "PDETect",
# 	}
# 	acq_mode_dict = {
# 	0 : "RTIMe",
# 	1 : "ETIMe",
# 	3 : "PDETect",
# 	}
# 	coupling_dict = {
# 	0 : "AC",
# 	1 : "DC",
# 	2 : "DCFIFTY",
# 	3 : "LFREJECT",
# 	}
# 	units_dict = {
# 	0 : "UNKNOWN",
# 	1 : "VOLT",
# 	2 : "SECOND",
# 	3 : "CONSTANT",
# 	4 : "AMP",
# 	5 : "DECIBEL",
# 	}
# 	preamble_string = do_query_string(":WAVeform:PREamble?")
# 	(
# 	wav_form, acq_type, wfmpts, avgcnt, x_increment, x_origin,
# 	x_reference, y_increment, y_origin, y_reference, coupling,
# 	x_display_range, x_display_origin, y_display_range,
# 	y_display_origin, date, time, frame_model, acq_mode,
# 	completion, x_units, y_units, max_bw_limit, min_bw_limit
# 	) = preamble_string.split(",")
# 	print("Waveform format: %s" % wav_form_dict[int(wav_form)])
# 	print("Acquire type: %s" % acq_type_dict[int(acq_type)])
# 	print("Waveform points desired: %s" % wfmpts)
# 	print("Waveform average count: %s" % avgcnt)
# 	print("Waveform X increment: %s" % x_increment)
# 	print("Waveform X origin: %s" % x_origin)
# 	print("Waveform X reference: %s" % x_reference) # Always 0.
# 	print("Waveform Y increment: %s" % y_increment)
# 	print("Waveform Y origin: %s" % y_origin)
# 	print("Waveform Y reference: %s" % y_reference) # Always 0.
# 	print("Coupling: %s" % coupling_dict[int(coupling)])
# 	print("Waveform X display range: %s" % x_display_range)
# 	print("Waveform X display origin: %s" % x_display_origin)
# 	print("Waveform Y display range: %s" % y_display_range)
# 	print("Waveform Y display origin: %s" % y_display_origin)
# 	print("Date: %s" % date)
# 	print("Time: %s" % time)
# 	print("Frame model #: %s" % frame_model)
# 	print("Acquire mode: %s" % acq_mode_dict[int(acq_mode)])
# 	print("Completion pct: %s" % completion)
# 	print("Waveform X units: %s" % units_dict[int(x_units)])
# 	print("Waveform Y units: %s" % units_dict[int(y_units)])
# 	print("Max BW limit: %s" % max_bw_limit)
# 	print("Min BW limit: %s" % min_bw_limit)
# 	# Get numeric values for later calculations.
# 	x_increment = do_query_number(":WAVeform:XINCrement?")
# 	x_origin = do_query_number(":WAVeform:XORigin?")
# 	y_increment = do_query_number(":WAVeform:YINCrement?")
# 	y_origin = do_query_number(":WAVeform:YORigin?")
# 	# Get the waveform data.
# 	do_command(":WAVeform:STReaming OFF")
# 	sData = do_query_ieee_block(":WAVeform:DATA?")
# 	# Unpack signed byte data.
# 	import numpy as np
# 	values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
# 	print("Number of data values: %d" % len(values))
# 	# Save waveform data values to CSV file.
# 	f = open("waveform_data.csv", "w")
# 	for i in range(0, len(values) - 1):
# 		time_val = x_origin + (i * x_increment)
# 		voltage = (values[i] * y_increment) + y_origin
# 		f.write("%E, %f\n" % (time_val, voltage))
# 	f.close()
# 	print("Waveform format BYTE data written to waveform_data.csv.")
# 	
# 	
			
class keysight_dsos054a:
	""" 
	connects to and downloads data from they Keysight DSOS054A oscilloscope 
	
	Notes
	-----
	*  Programming manual : https://www.keysight.com/upload/cmc_upload/All/Infiniium_prog_guide.pdf
	*  This code is based on snippets in the keysight programming manual
		
	"""
	
	def __init__(	self,
					address="TCPIP0::192.168.0.246::inst0::INSTR",
					timeout=20000,
					debug=False):
		self.address=address
		self.timeout=timeout
		self.init_connection()
		self.debug=False
		
	def check_instrument_errors(self, command):
		while True:
			error_string = self.Infiniium.query(":SYSTem:ERRor? STRing")
			if error_string: # If there is an error string value.
				if error_string.find("0,", 0, 2) == -1: # Not "No error".
					print("ERROR: %s, command: '%s'" % (error_string, command))
					print("Exited because of error.")
					sys.exit(1)
				else: # "No error"
					break
			else: # :SYSTem:ERRor? STRing should always return string.
				print("ERROR: :SYSTem:ERRor? STRing returned nothing, command: '%s'"% command)
				print("Exited because of error.")
				sys.exit(1)
		
	def do_command(self, command, hide_params=False):
		if hide_params:
			(header, data) = command.split(" ", 1)
		if self.debug:
			print("\nCmd = '%s'" % header)
		else:
			if self.debug:
				print("\nCmd = '%s'" % command)
		self.Infiniium.write("%s" % command)
		if hide_params:
			self.check_instrument_errors(header)
		else:
			self.check_instrument_errors(command)
		
	def do_command_ieee_block(self, command, values):
		""" Send a command and binary values and check for errors: """
		if self.debug:
			print("Cmb = '%s'" % command)
		self.Infiniium.write_binary_values("%s " % command, values, datatype='B')
		self.check_instrument_errors(command)
		
	def do_query_number(self,query):
		""" Send a query, check for errors, return floating-point value: """
		if self.debug:
			print("Qyn = '%s'" % query)
		results = self.Infiniium.query("%s" % query)
		self.check_instrument_errors(query)
		return float(results)
		
	def do_query_ieee_block(self,query,data_points=0):
		""" Send a query, check for errors, return binary values: """
		if self.debug:
			print("Qyb = '%s'" % query)
		result = self.Infiniium.query_binary_values("%s" % query, datatype='s',data_points=data_points)
		self.check_instrument_errors(query)
		return result#[0]
		
	def do_query_string(self,query):
		""" Send a query, check for errors, return string: """
		if self.debug:
			print("Qys = '%s'" % query)
		result = self.Infiniium.query("%s" % query)
		self.check_instrument_errors(query)
		return result
			
	def init_connection(self): #TODO print a connection confirmation
		""" initialize connection with unit """
		rm = pyvisa.ResourceManager()
		Infiniium = rm.open_resource(self.address)
		Infiniium.timeout = self.timeout
		Infiniium.clear()
		self.Infiniium=Infiniium
		
	def get_data_v2(self,ch_num=2,plot=False):
		""" 
		get data from scope 
		"""
		print("work in progress")
		# Clear status.
		self.do_command("*CLS")
		# Get and display the device's *IDN? string.
		idn_string = self.do_query_string("*IDN?")
		print("Identification string: '%s'" % idn_string)
		self.do_command(":MEASure:SOURce CHANnel%s"%ch_num)
		print(self.do_query_string(":MEASure:SOURce?"))
		
		
		# Get the waveform type.
		qresult = self.do_query_string(":WAVeform:TYPE?")
		print("Waveform type: %s" % qresult)
		# Get the number of waveform points.
		qresult = self.do_query_string(":WAVeform:POINts?")
		print("Waveform points: %s" % qresult)
		# Set the waveform source.
		self.do_command(":WAVeform:SOURce CHANnel%s"%ch_num)
		print("Waveform source:  "+self.do_query_string(":WAVeform:SOURce?"))
	# 	print("Waveform source: %s" % qresult)
		# Choose the format of the data returned:
		self.do_command(":WAVeform:FORMat BYTE")
		print("Waveform format: %s" % self.do_query_string(":WAVeform:FORMat?"))
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		# Get the waveform data.
		self.do_command(":WAVeform:STReaming OFF")
		sData = self.do_query_ieee_block(":WAVeform:DATA?")
		# Unpack signed byte data.
	# 	import numpy as np
		values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
	
	
		data = np.array(values) * y_increment + y_origin
		time = np.arange(len(data)) * x_increment + x_origin
	
		import xarray as xr
		data=xr.DataArray(	data,
								 dims='t',
								 coords=[time])
		
		if plot==True:
			fig,ax=plt.subplots()
			data.plot(ax=ax)
			
		return data
	
	
	def get_data_v3(self,ch_num=2,plot=False):
		""" 
		get data from scope 
		"""
		print("work in progress")
		# Clear status.
		self.do_command("*CLS")
# 		# Get and display the device's *IDN? string.
# 		idn_string = self.do_query_string("*IDN?")
# 		print("Identification string: '%s'" % idn_string)
		self.do_command(":MEASure:SOURce CHANnel%s"%ch_num)
		print('source: ', end='')
		print(self.do_query_string(":MEASure:SOURce?"))
		
		
		# Get the waveform type.
		qresult = self.do_query_string(":WAVeform:TYPE?")
		print("Waveform type: %s" % qresult)
		# Get the number of waveform points.
		qresult = self.do_query_string(":WAVeform:POINts?")
		print("Waveform points: %s" % qresult)
		# Set the waveform source.
		self.do_command(":WAVeform:SOURce CHANnel%s"%ch_num)
		print("Waveform source:  "+self.do_query_string(":WAVeform:SOURce?"))
	# 	print("Waveform source: %s" % qresult)
		# Choose the format of the data returned:
		self.do_command(":WAVeform:FORMat BYTE")
		print("Waveform format: %s" % self.do_query_string(":WAVeform:FORMat?"))
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		# Get the waveform data.
		self.do_command(":WAVeform:STReaming OFF")
		sData = self.do_query_ieee_block(":WAVeform:DATA?")
		# Unpack signed byte data.
	# 	import numpy as np
		values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
	
	
		data = np.array(values) * y_increment + y_origin
		time = np.arange(len(data)) * x_increment + x_origin
	
		import xarray as xr
		data=xr.DataArray(	data,
								 dims='t',
								 coords=[time])
		
		if plot==True:
			fig,ax=plt.subplots()
			data.plot(ax=ax)
			
		return data
	
	def get_all_data(self, plot=False):
		""" download all four channels of data """
		
		da1=self.get_data_v2(1)
		da2=self.get_data_v2(2)
		da3=self.get_data_v2(3)
		da4=self.get_data_v2(4)
		ds=xr.Dataset({'ch1':da1,'ch2':da2,'ch3':da3,'ch4':da4})
		
		if plot==True:
			fig,ax=plt.subplots()
			ds['ch1'].plot(ax=ax,label='ch1')
			ds['ch2'].plot(ax=ax,label='ch2')
			ds['ch3'].plot(ax=ax,label='ch3')
			ds['ch4'].plot(ax=ax,label='ch4')
			ax.legend()
		
		return ds	
	
	def set_SINGLE(self):
		self.do_command('SINGle')
		
	def set_STOP(self):
		self.do_command('STOP')
	
	def set_RUN(self):
		self.do_command('RUN')
	
	


		

if __name__ == "__main__":

	oscope=keysight_dsos054a()
	data=oscope.get_all_data(plot=True)
