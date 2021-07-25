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
	*  https://www.keysight.com/us/en/assets/9018-07141/programming-guides/9018-07141.pdf?success=true
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
		
	# %% setup scope
	
	def setup_segemented_data(self, points=50000, num_acq=1024):
		print("work in progress")
		
		# turn on segmented acquisition and specify number of poitns
		self.setup_acq(mode='SEGMented', points=points)
		
		# Turns on time tags (TTAGs), meaning the unit records the time of each acquisition
		self.do_command(':ACQuire:SEGMented:TTAGs ON')
		
		# set the number of segmented acquisitions
		self.do_command('ACQuire:SEGMented:COUNt %d' % num_acq)
		
# 		# Configures the unit to conveniently return all segmented data with one query
# 		self.Infiniium.query(':WAVeform:SEGMented:ALL ON')
	
	def setup_ch(self, ch_num=1, on=True, y_scale=0.1, y_offset=0, y_probe_atten_factor=1.0, coupling='DC', impedance='1M'):
		
		if on is True:
			self.do_command(':CHANnel%d:DISPlay ON' % ch_num)
			if coupling == 'DC' and impedance == '1M':
				self.do_command(":CHANnel%d:INPut %s" % (ch_num, 'DC'))
			elif coupling == 'DC' and impedance == '50':
				self.do_command(":CHANnel%d:INPut %s" % (ch_num, 'DC50'))
			elif coupling == 'AC':
				self.do_command(":CHANnel%d:INPut %s" % (ch_num, 'AC'))
			else:
				print("coupling and impedance combination not recognized")
			self.do_command(":CHANnel%d:SCALe %.3e" % (ch_num, y_scale))
			self.do_command(":CHANnel%d:OFFSet %.3e" % (ch_num, y_offset))
			self.do_command(":CHANnel%d:PROBe %.1e" % (ch_num, y_probe_atten_factor))
		else:
			self.do_command(':CHANnel%d:DISPlay OFF' % ch_num)
		
	def setup_time(self, t_scale=200e-6, t_offset=0):
		# Set horizontal scale and offset.
		self.do_command(":TIMebase:SCALe %.3e" % t_scale)
		self.do_command(":TIMebase:POSition % #.3e" % t_offset)
		
	def setup_acq(self, mode='RTIMe', points=32000, acq_rate=10e9):
		# Set the acquisition mode.
		self.do_command(":ACQuire:MODE %s" % mode)
		
		# Set the desired number of waveform points,
		self.do_command(":ACQuire:POINts %d" % points)
		
		# Set the acquisition rate
		self.do_command(":ACQuire:SRATe:ANALog %.3e" % acq_rate)
		
		
		# and capture an acquisition.
		# do_command(":DIGitize")
	
	def setup_trigger(self, ch_num=1, mode='EDGE', level=1e0, slope='POSitive'):
		
		self.do_command(":TRIGger:MODE %s" % mode)
		self.do_command(":TRIGger:EDGE:SOURce CHANnel%d" % ch_num)
		self.do_command(":TRIGger:LEVel CHANnel%d,%.3f" % (ch_num, level))
		self.do_command(":TRIGger:EDGE:SLOPe %s" % slope)
		
# 	# Get oscilloscope setup from *LRN? string.
# 	values_list = Infiniium.query_binary_values("*LRN?", datatype='s')
# 	check_instrument_errors()
# 	learn_bytes = values_list[0]
# 	# Set up oscilloscope by loading previously saved setup.
# 	f = open("setup_lrn.set", "rb")
# 	lrn_bytes = f.read()
# 	f.close()
		
	# %% Send commands and queries
		
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
	
	def do_query_ieee_block_UI1(self, query):
		result = self.do_query_ieee_block(query)
		self.check_instrument_errors(query)
		return result
	
	
	# %% save/load oscope setup
	
	def load_oscope_setup(self, setup_file="setup.stp"):
		# https://www.keysight.com/us/en/assets/9018-07141/programming-guides/9018-07141.pdf?success=true 
		# page 1789
		# Or, configure by loading a previously saved setup.
		f = open(setup_file, "rb")
		setup_bytes = f.read()
		f.close()
		import array
		self.do_command_ieee_block(":SYSTem:SETup", array.array('B', setup_bytes))
		print("Setup bytes restored: %d" % len(setup_bytes))

	def save_oscope_setup(self, setup_file="setup.stp"):
		# https://www.keysight.com/us/en/assets/9018-07141/programming-guides/9018-07141.pdf?success=true 
		# page 1788
		# Save oscilloscope setup.
		setup_bytes = self.do_query_ieee_block_UI1(":SYSTem:SETup?")
		nLength = len(setup_bytes)
		f = open(setup_file, "wb")
		f.write(bytearray(setup_bytes))
		f.close()
		print("Setup bytes saved: %d" % nLength)
	
	# %% connections
			
	def init_connection(self): #TODO print a connection confirmation
		""" initialize connection with unit """
		rm = pyvisa.ResourceManager()
		Infiniium = rm.open_resource(self.address)
		Infiniium.timeout = self.timeout
		Infiniium.clear()
		self.Infiniium=Infiniium
		
	def close_connection(self):
		self.Infiniium.close()
		
	# %% get data
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
	
	#TODO implement many of the functions described here:
	# https://www.keysight.com/us/en/assets/9018-07141/programming-guides/9018-07141.pdf?success=true
	# on page 1789 (before and after)
	
	def get_preamble(self):
		wav_form_dict = {
		0 : "ASCii",
		1 : "BYTE",
		2 : "WORD",
		3 : "LONG",
		4 : "LONGLONG",
		}
		acq_type_dict = {
		1 : "RAW",
		2 : "AVERage",
		3 : "VHIStogram",
		4 : "HHIStogram",
		6 : "INTerpolate",
		10 : "PDETect",
		}
		acq_mode_dict = {
		0 : "RTIMe",
		1 : "ETIMe",
		3 : "PDETect",
		}
		coupling_dict = {
		0 : "AC",
		1 : "DC",
		2 : "DCFIFTY",
		3 : "LFREJECT",
		}
		units_dict = {
		0 : "UNKNOWN",
		1 : "VOLT",
		2 : "SECOND",
		3 : "CONSTANT",
		4 : "AMP",
		5 : "DECIBEL",
		}
		preamble_string = self.do_query_string(":WAVeform:PREamble?")
		(
		wav_form, acq_type, wfmpts, avgcnt, x_increment, x_origin,
		x_reference, y_increment, y_origin, y_reference, coupling,
		x_display_range, x_display_origin, y_display_range,
		y_display_origin, date, time, frame_model, acq_mode,
		completion, x_units, y_units, max_bw_limit, min_bw_limit
		) = preamble_string.split(",")

		print("Waveform format: %s" % wav_form_dict[int(wav_form)])
		print("Acquire type: %s" % acq_type_dict[int(acq_type)])
		print("Waveform points desired: %s" % wfmpts)
		print("Waveform average count: %s" % avgcnt)
		print("Waveform X increment: %s" % x_increment)
		print("Waveform X origin: %s" % x_origin)
		print("Waveform X reference: %s" % x_reference) # Always 0.
		print("Waveform Y increment: %s" % y_increment)
		print("Waveform Y origin: %s" % y_origin)
		print("Waveform Y reference: %s" % y_reference) # Always 0.
		print("Coupling: %s" % coupling_dict[int(coupling)])
		print("Waveform X display range: %s" % x_display_range)
		print("Waveform X display origin: %s" % x_display_origin)
		print("Waveform Y display range: %s" % y_display_range)
		print("Waveform Y display origin: %s" % y_display_origin)
		print("Date: %s" % date)
		print("Time: %s" % time)
		print("Frame model #: %s" % frame_model)
		print("Acquire mode: %s" % acq_mode_dict[int(acq_mode)])
		print("Completion pct: %s" % completion)
		print("Waveform X units: %s" % units_dict[int(x_units)])
		print("Waveform Y units: %s" % units_dict[int(y_units)])
		print("Max BW limit: %s" % max_bw_limit)
		print("Min BW limit: %s" % min_bw_limit)
	
	
	def get_data_v4(self, ch_num=1, verbose=True, plot=False):
		
		# TODO(Jack) - The code only grabs the data that is visible on the screen instead of all data that is recorded.  Fix this.
		# TODO - What is the difference between :WAVeform:SOURce CHANnel and :MEASure:SOURce CHANnel ??
		# TODO How do I make this a command?
		
		# Clear status.
		self.do_command("*CLS")
		
		# selects the source for measurements.  Two channel numbers are provided and should be the same because we're not taking a "delta" measurement
		self.do_command(":MEASure:SOURce CHANnel%s" % ch_num) 
		if verbose:
			print(self.do_query_string(":MEASure:SOURce?"))
			
		# check on waveform type. RAW is the default (not averaged, interpolated, etc)
		# scope.do_command(":WAVeform:TYPE RAW") - query only command.  
		if verbose:
			print("Waveform type: %s" % self.do_query_string(":WAVeform:TYPE?"))
			
		# The :WAVeform:SOURce command selects a channel, function, waveform memory, or histogram as the waveform source.
		self.do_command(":WAVeform:SOURce CHANnel%s" % ch_num)
		if verbose:
			print("Waveform source:  " + self.do_query_string(":WAVeform:SOURce?"))

		#  This command controls how the data is formatted when it is sent from the oscilloscope, and pertains to all waveforms.
		self.do_command(":WAVeform:FORMat BYTE")  # note that BYTE is supposed to be very fast compared to other options
		if verbose:
			print("Waveform format: %s" % self.do_query_string(":WAVeform:FORMat?"))
		
		# get x and y resolution and offsets.  This is required to convert BYTES back to floats
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		
		# get data
		self.do_command(":WAVeform:STReaming OFF")
		sData = self.do_query_ieee_block(":WAVeform:DATA?")
		
		# convert data to floats
		values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
		data = np.array(values) * y_increment + y_origin
		time = np.arange(len(data)) * x_increment + x_origin
	
		# convert data to xr.DataArray
		import xarray as xr
		data=xr.DataArray(	data.astype(np.float32),
								 dims='t',
								 coords=[time])
		data.t.attrs = {'units': 's', 'long_name': 'Time'}
		data.attrs = {'units': 'V'}
		
		# optional plot
		if plot==True:
			fig,ax=plt.subplots()
			data.plot(ax=ax)
			
		return data
	
	def get_data_v5(self, ch_num=1, verbose=True, plot=False):
		
		# TODO(Jack) - The code only grabs the data that is visible on the screen instead of all data that is recorded.  Fix this.
		# TODO - What is the difference between :WAVeform:SOURce CHANnel and :MEASure:SOURce CHANnel ??
		# TODO How do I make this a command?
		
		# Clear status.
		self.do_command("*CLS")
		
		# selects the source for measurements.  Two channel numbers are provided and should be the same because we're not taking a "delta" measurement
		self.do_command(":MEASure:SOURce CHANnel%s" % ch_num) 
		if verbose:
			print(self.do_query_string(":MEASure:SOURce?"))
			
		# check on waveform type. RAW is the default (not averaged, interpolated, etc)
		# scope.do_command(":WAVeform:TYPE RAW") - query only command.  
		if verbose:
			print("Waveform type: %s" % self.do_query_string(":WAVeform:TYPE?"))
			
		# The :WAVeform:SOURce command selects a channel, function, waveform memory, or histogram as the waveform source.
		self.do_command(":WAVeform:SOURce CHANnel%s" % ch_num)
		if verbose:
			print("Waveform source:  " + self.do_query_string(":WAVeform:SOURce?"))

		#  This command controls how the data is formatted when it is sent from the oscilloscope, and pertains to all waveforms.
		self.do_command(":WAVeform:FORMat WORD")  # note that BYTE is supposed to be very fast compared to other options
		if verbose:
			print("Waveform format: %s" % self.do_query_string(":WAVeform:FORMat?"))
		
		# get x and y resolution and offsets.  This is required to convert BYTES back to floats
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		
		# get data
		self.do_command(":WAVeform:STReaming OFF")
# 		sData = self.do_query_ieee_block(":WAVeform:DATA?")
		values = self.Infiniium.query_binary_values(":WAVeform:DATA?", 'h', True) # https://docs.python.org/2/library/struct.html#format-characters # https://pyvisa.readthedocs.io/en/1.8/rvalues.html
		
		# convert data to floats
# 		values = struct.unpack("%db" % len(sData), np.array(sData).astype(np.byte))
		data = np.array(values) * y_increment + y_origin
		time = np.arange(len(data)) * x_increment + x_origin
	
		# convert data to xr.DataArray
		import xarray as xr
		data=xr.DataArray(	data.astype(np.float32),
								 dims='t',
								 coords=[time])
		data.t.attrs = {'units': 's', 'long_name': 'Time'}
		data.attrs = {'units': 'V'}
		
		# optional plot
		if plot==True:
			fig,ax=plt.subplots()
			data.plot(ax=ax)
			
		return data
	
	def get_data_v3(self, ch_num=2, plot=False):
		""" 
		get data from scope 
		"""
		print("work in progress")
		
		# TODO(Jack) - The code only grabs the data that is visible on the screen instead of all data that is recorded.  Fix this.
		# TODO - What is the difference between :WAVeform:SOURce CHANnel and :MEASure:SOURce CHANnel ??
		# Clear status.
		self.do_command("*CLS")
# 		# Get and display the device's *IDN? string.
# 		idn_string = self.do_query_string("*IDN?")
# 		print("Identification string: '%s'" % idn_string)
		self.do_command(":MEASure:SOURce CHANnel%s" % ch_num)
		print('source: ', end='')
		print(self.do_query_string(":MEASure:SOURce?"))
		
		# Get the waveform type.
		qresult = self.do_query_string(":WAVeform:TYPE?")
		print("Waveform type: %s" % qresult)
		# Get the number of waveform points.
		qresult = self.do_query_string(":WAVeform:POINts?")
		print("Waveform points: %s" % qresult)
		# Set the waveform source.
		self.do_command(":WAVeform:SOURce CHANnel%s" % ch_num)
		print("Waveform source:  " + self.do_query_string(":WAVeform:SOURce?"))
	# 	print("Waveform source: %s" % qresult)
		# Choose the format of the data returned:
		self.do_command(":WAVeform:FORMat BYTE")  # note that BYTE is supposed to be very fast compared to other options
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
		data.t.attrs = {'units': 's', 'long_name': 'Time'}
		data.attrs = {'units': 'V'}
		
		if plot==True:
			fig,ax=plt.subplots()
			data.plot(ax=ax)
			
		return data
	
	def get_segemented_data_v2(self, ch_num=1, plot=False):
		print("work in progress")
		
		# is this command needed?
		self.do_command(":WAVeform:SOURce CHANnel%s" % ch_num)
		
		self.do_command(":MEASure:SOURce CHANnel%s" % ch_num)
		count = int(self.do_query_number('WAVeform:SEGMented:COUNt?'))
		self.do_command('WAVeform:SEGMented:ALL ON')
# 		self.do_command(":WAVeform:FORMat BYT")
		
 		# get data
		values = self.Infiniium.query_binary_values(":WAVeform:DATA?", 'h', True) # https://docs.python.org/2/library/struct.html#format-characters # https://pyvisa.readthedocs.io/en/1.8/rvalues.html
		
# 		sData = np.array(self.do_query_ieee_block(":WAVeform:DATA?"), dtype=float)
# 		
# 		# unwrap data # TODO figure out why the data comes in like this and needs to be unwrapped
# 		if True:
# 			# fig, ax = plt.subplots()
# 			# ax.plot(sData[:150000], marker='.', linestyle='')
# 			ind = np.where(sData > 256/2) 
# 			sData[ind] -= 256
# 			# ax.plot(sData[:150000], marker='', linestyle='-')
# 			sData -= sData.mean()
# 		
		# get x and y offsets and scaling parameters
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		
		# apply scalings and offset to data and time
		data = np.array(values).astype(float) * y_increment + y_origin
		time = np.arange(int(len(values) / count)) * x_increment + x_origin
	
		# get time of each segement 
		# TODO this is brute force but the :WAVeform:SEGMented:XLISt? command doesn't work.  Fix?
		# TODO optionally, just grab the last TTAG, divide by (N-1) and use that as the dt value
		if True:
			seg_time = np.zeros(count, dtype=float)
			for i in range(count):
				self.do_command('ACQuire:SEGMented:INDex %d' % (i + 1))
				seg_time[i] = float(self.Infiniium.query(':WAVeform:SEGMented:TTAG?'))
		else:
			self.do_command('ACQuire:SEGMented:INDex 1024')
			T = self.do_query_number('WAVeform:SEGMented:TTAG?')
			dT = T / (count - 1)
			seg_time = np.arange(count, dtype=float) * dT
			
		# format data as an xarray.DataArray
		Data = xr.DataArray(data.reshape((count, -1)), dims=['seg_time', 'time'], coords=[seg_time, time])
		
		if plot is True:
			fig, ax = plt.subplots()
			for i in range(10):
				Data[i,:].plot(ax=ax, label='%.3e s' % float(Data.coords[Data.dims[0]][i].data))
			ax.legend()
			
		return Data
	
	def get_segemented_data(self, ch_num=1, plot=False):
		print("work in progress")
		
		# is this command needed?
		self.do_command(":WAVeform:SOURce CHANnel%s" % ch_num)
		
		self.do_command(":MEASure:SOURce CHANnel%s" % ch_num)
		count = int(self.do_query_number('WAVeform:SEGMented:COUNt?'))
		self.do_command('WAVeform:SEGMented:ALL ON')
# 		self.do_command(":WAVeform:FORMat BYT")
		
		# get data
		sData = np.array(self.do_query_ieee_block(":WAVeform:DATA?"), dtype=float)
		
		# unwrap data # TODO figure out why the data comes in like this and needs to be unwrapped
		if True:
			# fig, ax = plt.subplots()
			# ax.plot(sData[:150000], marker='.', linestyle='')
			ind = np.where(sData > 256/2) 
			sData[ind] -= 256
			# ax.plot(sData[:150000], marker='', linestyle='-')
			sData -= sData.mean()
		
		# get x and y offsets and scaling parameters
		x_increment = self.do_query_number(":WAVeform:XINCrement?")
		x_origin = self.do_query_number(":WAVeform:XORigin?")
		y_increment = self.do_query_number(":WAVeform:YINCrement?")
		y_origin = self.do_query_number(":WAVeform:YORigin?")
		
		# apply scalings and offset to data and time
		data = np.array(sData).astype(float) * y_increment + y_origin
		time = np.arange(int(len(sData) / count)) * x_increment + x_origin
	
		# get time of each segement 
		# TODO this is brute force but the :WAVeform:SEGMented:XLISt? command doesn't work.  Fix?
		# TODO optionally, just grab the last TTAG, divide by (N-1) and use that as the dt value
		if False:
			seg_time = np.zeros(count, dtype=float)
			for i in range(count):
				self.do_command('ACQuire:SEGMented:INDex %d' % (i + 1))
				seg_time[i] = float(self.Infiniium.query(':WAVeform:SEGMented:TTAG?'))
		else:
			self.do_command('ACQuire:SEGMented:INDex 1024')
			T = self.do_query_number('WAVeform:SEGMented:TTAG?')
			dT = T / (count - 1)
			seg_time = np.arange(count, dtype=float) * dT
			
		# format data as an xarray.DataArray
		Data = xr.DataArray(data.reshape((count, -1)), dims=['seg_time', 'time'], coords=[seg_time, time])
		
		if plot is True:
			fig, ax = plt.subplots()
			for i in range(10):
				Data[i,:].plot(ax=ax, label='%.3e s' % float(Data.coords[Data.dims[0]][i].data))
			ax.legend()
			
		return Data
	
	def get_all_data(self, chs=[1, 2, 3, 4], plot=False):
		""" download all four channels of data """
		
		ds = xr.Dataset()
		
		for ch in chs:
			da = self.get_data_v5(ch)
			if list(ds.keys()) == []:
				ds['ch%d' % ch] = xr.DataArray(da.data, dims=da.dims, coords=[da.coords['t'].data])
			else:
				key1=list(ds.keys())[0]
				ds['ch%d' % ch] = xr.DataArray(da.data, dims=ds[key1].dims, coords=[ds[key1].coords['t'].data])

		if plot is True:
			fig, ax = plt.subplots()
			for key in list(ds.keys()):
				ds[key].plot(ax=ax, label=key)
			ax.legend()
			
		return ds	
	
	def get_all_segmented_data(self, chs=[1, 2, 3, 4], plot=False):
		""" download all four channels of data """
		
		ds = xr.Dataset()
		
		for ch in chs:
			da = self.get_segemented_data_v2(ch, plot=plot)
			if list(ds.keys()) == []:
				ds['ch%d' % ch] = xr.DataArray(da.data, dims=da.dims, coords=da.coords)
			else:
				key1=list(ds.keys())[0]
				ds['ch%d' % ch] = xr.DataArray(da.data, dims=ds[key1].dims, coords=ds[key1].coords)

# 		if plot is True:
# 			fig, ax = plt.subplots()
# 			for key in list(ds.keys()):
# 				ds[key].plot(ax=ax, label=key)
# 			ax.legend()
			
		return ds	
	
	
	def get_data_multichannel(self, chs=[1, 2, 3, 4], ch_names=[], scaling=[], plot=False):
		""" download all four channels of data """
		
		# stop scope
		self.set_STOP()
		
		# grab data from each channel.  Optionally apply scaling factor
		ds = xr.Dataset()
		for i, ch in enumerate(chs):
			print(i, ch)
			data = self.get_data_v3(ch)
			if scaling != []:
				data *= scaling[i]
			if ch_names == []:
				data.name = 'ch{}'.format(ch)
			else:
				data.name = ch_names[i]
			ds[data.name] = data
	
		# optional plot
		if plot is True:
			keys = list(ds.keys())
			fig, ax = plt.subplots(len(keys), sharex=True)
			for i, key in enumerate(keys):
				ds[key].plot(ax=ax[i], label=ds[key].name)
				ax[i].legend()
			
		# restart scope
		self.set_RUN()
		
		return ds	
	
	# %% set oscope operation
	
	def set_SINGLE(self):
		self.do_command('SINGle')
		
	def set_STOP(self):
		self.do_command('STOP')
	
	def set_RUN(self):
		self.do_command('RUN')
	
	
# %% main

if __name__ == "__main__":
	
	if False:
		
		oscope = keysight_dsos054a()
		oscope.get_data_v5(plot=True)
		
	if False:
		
		oscope = keysight_dsos054a()
		data = oscope.get_segemented_data_v2(plot=True)
		
	if False:
		# open connection
		oscope = keysight_dsos054a()
		
		# setup time
		oscope.setup_time(t_scale=500e-9, t_offset=0)
		
		# setup acq
		oscope.setup_acq(mode='RTIMe', points=50000)

		# setup channels
		oscope.setup_ch(ch_num=1, y_scale=0.5, y_offset=0, y_probe_atten_factor=1.0)
		oscope.setup_ch(ch_num=2, y_scale=0.5, y_offset=0, y_probe_atten_factor=1.0)
		oscope.setup_ch(ch_num=3, y_scale=0.5, y_offset=0, y_probe_atten_factor=1.0)
		oscope.setup_ch(ch_num=4, on=False)
		
		# setup trigger
		oscope.setup_trigger(ch_num=3, mode='EDGE', level=0.5, slope='POSitive')
		
		# get data
		data = oscope.get_all_data([1, 2, 3], plot=True)
		
		# close connection
		# oscope.close_connection()
		