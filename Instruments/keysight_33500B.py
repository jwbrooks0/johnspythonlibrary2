# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:14:17 2021

@author: jwbrooks

References
----------
 * https://github.com/avenkatramani/programs/blob/489ce6e9944b2ef8f19c6aa138df2f792c73c930/agilent33500B.py
 * https://github.com/gabrielfedel/py4syn-old/blob/6653faa788b273c8a592ae7548f9027fd95cc62a/py4syn/epics/Keysight33500BClass.py
 * from qcodes.instrument_drivers.Keysight import KeysightAgilent_33XXX
"""

# import johnspythonlibrary2 as jpl2
# import nrl_code as nrl
# import numpy as np
# import matplotlib.pyplot as plt
# from time import sleep

from johnspythonlibrary2.Instruments import instr_tools


class keysight_33500B:
	
	"""
	References
	----------
	* https://literature.cdn.keysight.com/litweb/pdf/33500-90901.pdf
	"""
	
	def __init__(self, address=''):
		
		if address=='':
			address = instr_tools.find_device('?*335?*')[0]
		self.address=address
		self.init_connection()
		self.output_on(False, 1)
		self.output_on(False, 2)
		

	def init_connection(self):
		
		session, resourceManager = instr_tools.connect_to_device(self.address)
		self.session = session
		self.resourceManager = resourceManager
		
	def DC(self,V=0, ch=1):
		""" set output to DC mode at voltage V """
		
		self.session.write(":SOUR%d:APPL:DC DEF,DEF, %.4f" % (ch,V))
		
	def output_on(self, on=True, ch=1):
		if on==True:
			self.session.write(":OUTP%d ON" % ch)
		else:
			self.session.write(":OUTP%d OFF" % ch)
		
	def fixed_freq_sine_wave(self, f=1e3, v_pp=0.1, v_offset=0.0, ch=1, phase_deg=0):
		self.session.write('SOURce{:d}:FUNCtion {}'.format(ch,'SIN') )
		self.session.write('SOURce{:d}:FREQuency:MODE {}'.format(ch,'CW') )
		self.session.write('SOURce{:d}:FREQuency {:.4f}'.format(ch,f) )
		self.session.write('SOURce{:d}:VOLTage {:.4f}'.format(ch,v_pp) )
		self.session.write('SOURce{:d}:VOLTage:OFFSet {:.4f}'.format(ch,v_offset) )
		self.session.write('SOURce{:d}:PHASe {:.4f}'.format(ch, phase_deg) )
		
					  
if __name__=='__main__':
	siggen=keysight_33500B()
	siggen.fixed_freq_sine_wave( f=1e3, v_pp=4, v_offset=0, ch=1, phase_deg=0)
# 	siggen.output_on()
# 	siggen.output_on(False)