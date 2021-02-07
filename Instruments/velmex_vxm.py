# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:51:40 2021

@author: jwbrooks
"""

# import johnspythonlibrary2 as jpl2
# import numpy as np
# import matplotlib.pyplot as plt
import serial
import time

class velmex_vxm:
	
	"""
	
	TODO : Test this program
	
	Notes
	-----
	* Based on code from here : https://github.com/unkatoco/Velmex/blob/6443294f597839bdb9d9b18d2b47abb6c4e745e8/velmex.py
	"""
	
	def __init__(self, port = 0):
		
		self.instrument= serial.Serial(port=port, baudrate=9600, bytesize=8 , parity='E', stopbits=1, timeout=.1) # vxm
		self.instrument.write("F")  # enable online mode
		self.instrument.write("C")  # clear current program
		
	def _move_motor(self, command):
	    self.instrument.write("C")  # clear current program
	    self.instrument.write(command) # send movement command
	    self.instrument.write("R")  # run current program
		
	def move_by (self, motor_name, distance):
	    self._move_motor("I" + str(motor_name) + "M" + str(distance) + ",") # send movement command
	
	def move_to (self, motor_name, destination):
	    self._move_motor("IA" + str(motor_name) + "M" + str(destination) + ",") # send movement command
		
	def home_motor(self, motor_name):
	    self._move_motor("IA" + str(motor_name) + "M0" + ",") # send movement command
		
	def wait_until_done(self):
		""" delay until current program is done    """
		
		self.instrument.readline()                  # clear current buffer
		self.instrument.write('V')                  # query for velmex's status 
		while (self.instrument.readline()=="B"):    # if busy, loop until not busy
			time.sleep(0.01)
			self.instrument.write("V")              

		    