
try:
	import pyvisa as visa
	
	from .keithley_24xx import keithley_24xx
	from .keysight_dsos054a import keysight_dsos054a
	from .agilent_E5061B import agilent_E5061B
	from .keysight_n5700 import keysight_n5700
	from .keysight_33500B import keysight_33500B
	from .velmex_vxm import velmex_vxm
	
	
except: 
	print("Failed to load the Instruments sublibrary.  Do you have the pyvisa and pyserial libraries installed?")
