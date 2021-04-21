

import pyvisa as visa
from johnspythonlibrary2.Instruments import instr_tools

class sorensenPower(object):
	"""
	
	References
	----------
	* Based on the code found here : https://github.com/ryandavid/sorensen-power/blob/master/sorensenPower.py
	* Programming manual : https://www.powerandtest.com/-/media/ametekprogrammablepower/files/dc-power-supplies/sg-series/manuals/programming-ethernet-ieee-rs232-rev-m/sga.pdf?dmc=1&la=en&revision=9ff4e4aa-8b04-4f6d-baec-83cf6fdf38f6&rev=1617306336850
	* Operating manual : https://www.powerandtest.com/-/media/ametekprogrammablepower/files/dc-power-supplies/sg-series/manuals/operations-manual/sga-series_operation_manual_m550129-01_rah.pdf?dmc=1&la=en&revision=0ce05236-c9a2-4c58-9946-5d68db60f3a4&rev=1617306336850
	"""
	DEFAULT_TIMEOUT = 0.125

	def __init__(self, address='TCPIP0::192.168.0.244::inst0::INSTR', timeout=DEFAULT_TIMEOUT, debug=False):
		self.address = address
		self.timeout = timeout
		self.connect()
		
	def connect(self):
		""" initialize connection with unit """
		rm = visa.ResourceManager()
		self.instrument = rm.open_resource(self.address)
		self.instrument.timeout = self.timeout
		self.device_details=self.instrument.query("*IDN?")
		print("Connected to '%s' at '%s'" % (self.device_details, self.address))
			
				
	def do_query_string(self,query):
		""" Send a query, check for errors, return string: """
		result = self.instrument.query("%s" % query)
		return result

	def __del__(self):
		# Make sure we return control to local.
		self.disconnect()

	def _writeCommand(self, command):
		self.instrument.write("%s" % command)

	def disconnect(self, returnToLocal=True):
		self.setOutputCurrent(0)
		self.setOutputVoltage(0)
		self._writeCommand(":SYST:LOCAL ON")
		self.instrument.close()

	def getOutputVoltage(self):
		
		voltageASCII = self.do_query_string(":MEAS:VOLT?")
		voltage = float(voltageASCII.strip())

		return voltage

	def getOutputCurrent(self):
		currentASCII = self.do_query_string(":MEAS:CURR?")
		current = float(currentASCII.strip())

		return current
	
	def measure_volt_and_current(self):
		V=self.getOutputVoltage()
		I=self.getOutputCurrent()
		
		return V,I

	def setOutputVoltage(self, voltage):
		self.do_query_string(":SOUR:VOLT {:1.03f}".format(voltage))

	def setOutputCurrent(self, current):
		self.do_query_string(":SOUR:CURR {:1.03f}".format(current))
		
	def setZeroOutput(self):
		self.setOutputCurrent(0)
		self.setOutputVoltage(0)
		
		

	def getStatus(self):
		statusASCII = self.do_query_string(':SOUR:STAT:BLOC?').strip().split(',')
		print(statusASCII)
# 		# This command returns 25 pieces of data.  The original code was written for 23.  I can't figure out the manual to determine what is what.  

# 		if (len(statusASCII) == 23):
# 			statusRegister = int(statusASCII[3])
# 			overTemperature = bool((statusRegister >> 4) & 0x01)
# 			overVoltage	 = bool((statusRegister >> 3) & 0x01)
# 			constantCurrent = bool((statusRegister >> 1) & 0x01)
# 			constantVoltage = bool((statusRegister >> 0) & 0x01)

# 			status = {
# 				'channelNumber'	 : int(statusASCII[0]),
# 				'onlineStatus'	  : int(statusASCII[1]),
# 				'statusFlags'	   : int(statusASCII[2]),
# 				'statusRegister'	: statusRegister,
# 				'accumulatedStatus' : int(statusASCII[4]),
# 				'faultMask'		 : int(statusASCII[5]),
# 				'faultRegister'	 : int(statusASCII[6]),
# 				'errorRegister'	 : int(statusASCII[7]),
# 				'overTemperature'   : overTemperature,
# 				'overVoltage'	   : overVoltage,
# 				'constantCurrent'   : constantCurrent,
# 				'constantVoltage'   : constantVoltage,
# 				'serialNumber'	  : statusASCII[8],
# 				'voltageCapability' : float(statusASCII[9]),
# 				'currentCapability' : float(statusASCII[10]),
# 				'overVoltage'	   : float(statusASCII[11]),
# 				'voltageDacGain'	: float(statusASCII[12]),
# 				'voltageDacOffset'  : float(statusASCII[13]),
# 				'currentDacGain'	: float(statusASCII[14]),
# 				'currentDacOffset'  : float(statusASCII[15]),
# 				'protectionDacGain' : float(statusASCII[16]),
# 				'protectionDacOffset': float(statusASCII[17]),
# 				'voltageAdcGain'	: float(statusASCII[18]),
# 				'voltageAdcOffset'  : float(statusASCII[19]),
# 				'currentAdcGain'	: float(statusASCII[20]),
# 				'currentAcOffset'   : float(statusASCII[21]),
# 				'model'			 : statusASCII[22]
# 			}

# 			self.model = status['model']
# 			self.serialNumber = status['serialNumber']
# 			self.maxCurrent = status['currentCapability']
# 			self.maxVoltage = status['voltageCapability']

# 		return status
	
if __name__ == "__main__":
	instrument=sorensenPower()