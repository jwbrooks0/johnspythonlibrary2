

# specs and performance : https://download.tek.com/manual/077248700web.pdf
# programming manual : https://download.tek.com/manual/077006105web.pdf
# service : file:///C:/Users/jwbrooks/Downloads/077030503web.pdf
# quick start : http://www.av.it.pt/Medidas/Data/Manuais%20&%20Tutoriais/67%20&%2068%20-%20AWG%205K%20&%207K/071185101.pdf


import time
import pyvisa as visa
import warnings
import socket
import numpy as _np

# if False:
# 	vna_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 	vna_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# 	vna_socket.settimeout(5.0)
# 	ip='192.168.0.229'
# 	port=5025
# 	vna_address = ((ip, port))
# 	vna_socket.connect(vna_address)
# vna_socket.send()

# # class tektronix_awg70xx:
# address = "TCPIP::192.168.0.229::5025::SOCKET"
# # address = 'TCPIP0::192.168.0.229::inst0::INSTR'
# rm = visa.ResourceManager()
# # print(rm.list_resources())
# instrument = rm.open_resource(address)
# # print('Connected to : ' + instrument.query("*IDN?"))
# instrument.send("*IDN?")
# # 
# import tek_awg
# tek_awg.TekAwg.connect_to_ip(ip=ip)
# # https://github.com/dahlend/TekAwg/blob/master/TekAwg.py


class awg7062B:
	# https://github.com/dahlend/TekAwg/blob/master/TekAwg.py
	
	def __init__(self, ip='192.168.0.229', port=5001, timeout=2000):
		
		self.timeout = timeout
		self.address = "TCPIP::{ip}::{port}::SOCKET".format(ip=ip, port=port)
		self.connect()
		
	def connect(self):
		rm=visa.ResourceManager()
		self.instrument = rm.open_resource(self.address, read_termination='\n', query_delay=1e-3, timeout=self.timeout)
		print("Connected to:", self.send_query("*IDN?"))
		print('at', self.address)
		
	def send_command(self, cmd):
		self.instrument.write(cmd)
		
	def send_query(self, cmd):
		return self.instrument.query(cmd)
	
	def send_query_binary_values(self, cmd):
		return self.instrument.query_binary_values(cmd)
	
	def close(self):
		self.instrument.close()

	def open(self):
		self.instrument.open()
		
	def wait_until_commands_executed(self):
		response = self.send_query('*OPC?')
		if response != '1':
			warnings.warn('Unexpected answer on "*OPC?": %s' % response)

	def _parse_waveform_name(self, waveform_name):
		return '"%s"' % waveform_name.strip().strip('"')
	
		
# 	def get_waveform_data(self, waveform_name):
# 		"""Get the raw waveform data from the AWG
#  			Args:
# 				waveform_name: Name of the waveform to get

#  			Returns: a string of binary containing the data from the AWG, header has been removed

#  			Raises:
# 				IOError if there was a timeout, most likely due to connection or incorrect name
# 		"""
# 		waveform_name = self._parse_waveform_name(waveform_name)

# 		wf_length = self.get_waveform_lengths(waveform_name)
# 		data_type = self.get_waveform_types(waveform_name)
# 		if data_type == 'REAL':
# 			dtype = Waveform.real_t.dtype
# 		else:
# 			dtype = Waveform.int_t.dtype

# 		n_chunks = (wf_length + chunk_size - 1) // chunk_size

# 		waveform_data_cmd = 'WLISt:WAVeform:DATA? %s,{start}, {size}' % waveform_name

# 		waveform_data = []

# 		remaining_points = wf_length
# 		for chunk in range(n_chunks):
#  			cmd = waveform_data_cmd.format(start=chunk*chunk_size, size=min(chunk_size, remaining_points))

#  			received = self.instrument.query_binary_values(cmd, datatype='s', container=tuple,
#  														   header_fmt='ieee')

#  			waveform_data.extend(received)
#  			remaining_points -= chunk_size

# 		waveform_data = b''.join(waveform_data)

# 		return Waveform.from_binary(np.frombuffer(waveform_data, dtype=dtype))
	
# 	def get_waveform_timestamps(self, waveform_name):
# 		"""Returns the creation/edit timestamp of waveforms which are stored on the AWG,

# 			Args:
# 				waveform_list: A single waveform name, or list of names

# 			Returns: list of strings containing date of creation or last edit

# 			Raises:
# 				IOError if fewer types were returned then asked for"""
# # 		if isinstance(waveform_name, (str, int)):
# # 			return self.get_waveform_timestamps([waveform_names])[0]

# # 		waveform_names = self._parse_waveform_names(waveform_names)

# 		return self.query_chunked(
# 			map(':WLIST:WAV:TST? {}'.format, waveform_name),
# 			expected_responses=len(waveform_names), chunk_size=16
# 		)

# 	def _get_chunked(iterable, chunk_size):
# 		iterator = iter(iterable)
# 		import itertools
# 	
# 		while True:
# 			chunk = itertools.islice(iterator, chunk_size)
# 	
# 			try:
# 				first_element = next(chunk)
# 			except StopIteration:
# 				return
# 	
# 			yield itertools.chain((first_element, ), chunk)

	def get_run_state(self):
		"""Gets the current state of the AWG, possible states are:
		stopped, waiting for trigger, or running"""
		state = self.send_query("AWGControl:RSTate?")
		if "0" in state:
			return "Stopped"
		elif "1" in state:
			return "Waiting for Trigger"
		elif "2" in state:
			return "Running"
		raise IOError("Not valid run state")
		
	def run(self):
		"""Start running the AWG"""
		self.send_command("AWGControl:RUN")

	def stop(self):
		"""Stop the AWG"""
		self.send_command("AWGCONTROL:STOP")
		
	def del_waveform(self, waveform_name):
		"""Delete Specified Waveform"""
		self.send_command('WLISt:WAVeform:DELete %s' % waveform_name)

# 	def new_waveform(self, waveform):
# 		"""

# 		Args:
# 			waveform_name:
# 			waveform:
# 			chunk_size: Default is 10KB

# 		Returns:

# 		"""
# # 		data_type = waveform.data_type
# # 		wf_length = waveform.size
# # 		data = waveform.binary

# # 		waveform_name = '"%s"' % waveform_name.replace('"', '').strip()

# 		name = '"hello4"'
# 		size = 1024
# 		data_type = 'INTEGER'
# 		awg.send_command('WLISt:WAVeform:NEW {name}, {size}, {data_type}'.format(name=name,
# 																		 size=size,
# 																		 data_type=data_type))
# 		awg.instrument.write_binary_values('WLIST:WAVEFORM:DATA {name},{offset},{size}'.format(name=name, offset=0, size=size), _np.linspace(0, 99,100).astype(int))
# # 		data_cmd = 'WLIST:WAVEFORM:DATA {name},{offset},{size},{data}'


	def get_waveform_names(self):
		"""Returns a list of all the currently saved waveforms on the AWG"""

		num_saved_waveforms = int(self.send_query("WLIST:SIZE?"))

		waveform_list_cmd = 'WLIST:'
		waveform_list_cmd += ";".join(["NAME? "+str(i) for i in range(0, num_saved_waveforms)])

		waveform_list = self.send_query(waveform_list_cmd).split(';')#, True, num_saved_waveforms).split(";")

		return waveform_list
	
	def set_channel_output(self, state='ON', channel=1):
		self.send_command('OUTput{ch}:STATe {st}'.format(ch=channel, st=state))
	
	def get_waveform_data(self, name='"*Square10"'):
		"""Get the raw waveform data from the AWG, this will be in the packed format containing
		both the channel waveforms as well as the markers, this needs to be correctly formatted.
			Args:
				filename: name of the file to get from the AWG
			Returns: a string of binary containing the data from the AWG, header has been removed
			Raises:
				IOError if there was a timeout, most likely due to connection or incorrect name
		"""

		return self.send_query_binary_values('WLISt:WAVeform:DATA? %s' % names[0])

	def set_sampling_rate(self, sampling_rate=6e9):
		self.send_command('SOURCE1:FREQUENCY %.3e' % sampling_rate)
		
	def set_waveform(self, name='"*Square10"'):
		self.send_command('SOURCE1:WAVEFORM %s' % name)
		
	def import_waveform_from_file(self, filename='"C:\Documents and Settings\OEM\Desktop\dgdt_1_10001.txt"'):
		name='"'+filename.split('\\')[-1].split('.')[0]+'"'
		self.send_command('MMEMORY:IMPORT {name},{filename},txt'.format(name=name, filename=filename))
		
		
if __name__ == '__main__':
	awg = awg7062B(timeout=1000)
	names = awg.get_waveform_names()
	awg.get_waveform_data(names[0])
	awg.import_waveform_from_file('"C:\Documents and Settings\OEM\Desktop\dgdt_1_10001.txt"')
	awg.set_waveform('"dgdt_1_10001"')
	awg.set_sampling_rate(sampling_rate=6e9)
	awg.run()
	awg.set_channel_output()
# 	names = awg.get_waveform_names()
# 	awg.del_waveform('"*DC"')
# 	names2 = awg.get_waveform_names()
# 	waveform_name = '*"Sine10"'
# 	waveform_data_cmd = 'WLISt:WAVeform:DATA? %s,{start}, {size}' % waveform_name
# 	waveform_data_cmd = waveform_data_cmd.format(start=0, size=1)
# # 	awg.instrument.query(waveform_data_cmd)
# 	cmd = waveform_data_cmd
# 	received = awg.instrument.query_binary_values(cmd, datatype='s', container=tuple,
#  														   header_fmt='ieee')
# 	awg.send_query(':WLIST:WAV:TST? "*Sine10"')
# 	awg.send_query("AWGControl:RSTate?")
# 	awg.send_command("AWGControl:RUN")
# 	print(awg.get_run_state())
# 	awg.run()
# 	print(awg.get_run_state())
# 	awg.stop()
# 	print(awg.get_run_state())
# 	awg.instrument.close()
# if False:
# 	import pyvisa as visa
# 	ip = '192.168.0.229'
# 	port = 5001
# 	address = 
# 	
# 	float(instrument.query("FREQ?"))
# 	instrument.write('*CLS\n')
# 	instrument.query("*ESR?\n")
# 	instrument.query("STATus:OPERation:CONDition?\n")
# 	datatype='s', container=tuple,												header_fmt='ieee'
	
	
# 	connect((ip, port))
# 	num_saved_waveforms = int(vna_socket.write("WLIST:SIZE?", True))
# 	
# 	waveform_list_cmd = 'WLIST:'
# 	waveform_list_cmd += ";".join(["NAME? "+str(i) for i in range(0, num_saved_waveforms)])
# 	
# 	waveform_list = self.write(waveform_list_cmd, True, num_saved_waveforms).split(";")
# 	
# 	return waveform_list
