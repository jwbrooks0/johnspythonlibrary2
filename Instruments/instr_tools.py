# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:43:59 2021

@author: jwbrooks

References
----------
 * https://www.keysight.com/main/editorial.jspx?cc=US&lc=eng&ckey=2798637&nid=-35489.384515&id=2798637
"""

# import johnspythonlibrary2 as jpl2
# import nrl_code as nrl
# import numpy as np
# import matplotlib.pyplot as plt


import pyvisa as visa
import sys


def list_all_devices():
	resourceManager = visa.ResourceManager()
	out = resourceManager.list_resources()
	resourceManager.close()
	
	print(out)
	
	return out

def find_device(partial_name='TCPIP?*'):
	"""
	
	# All instruments (no INTFC, BACKPLANE or MEMACC)
	find_device('?*INSTR')
	# PXI modules
	find_device('PXI?*INSTR')
	# USB devices
	find_device('USB?*INSTR')
	# GPIB instruments
	find_device('GPIB?*')
	# GPIB interfaces
	find_device('GPIB?*INTFC')
	# GPIB instruments on the GPIB0 interface
	find_device('GPIB0?*INSTR')
	# LAN instruments
	find_device('TCPIP?*')
	# SOCKET (::SOCKET) instruments
	find_device('TCPIP?*SOCKET')
	# VXI-11 (inst) instruments
	find_device('TCPIP?*inst?*INSTR')
	# HiSLIP (hislip) instruments
	find_device('TCPIP?*hislip?*INSTR')
	# RS-232 instruments
	find_device('ASRL?*INSTR')
	"""
	
	resourceManager = visa.ResourceManager()

	print('Find with search string \'%s\':' % partial_name)
	devices = resourceManager.list_resources(partial_name)
	if len(devices) > 0:
		for device in devices:
			print('\t%s' % device)
	else:
		print('... didn\'t find anything!')

	resourceManager.close()
	
	return devices


def connect_to_device(VISA_ADDRESS):
	
	"""
	
	Examples
	--------
	
	Example 1 ::
		dev_list = find_device()
		session, resourceManager = connect_to_device(dev_list[1])
		
	"""
		
	try:
	    # Create a connection (session) to the instrument
	    resourceManager = visa.ResourceManager()
	    session = resourceManager.open_resource(VISA_ADDRESS)
	except visa.Error as ex:
		print(ex)
		raise Exception('Couldn\'t connect to \'%s\', exiting now...' % VISA_ADDRESS)

	
	# For Serial and TCP/IP socket connections enable the read Termination Character, or read's will timeout
	if session.resource_name.startswith('ASRL') or session.resource_name.endswith('SOCKET'):
	    session.read_termination = '\n'
	
	# Send *IDN? and read the response
	session.write('*IDN?')
	idn = session.read()
	
	print('Successfully connected to \n%sat address\n%s'%(idn, VISA_ADDRESS))
	
	return session, resourceManager