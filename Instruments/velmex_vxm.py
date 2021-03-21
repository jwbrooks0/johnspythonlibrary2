# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:51:40 2021

@author: jwbrooks
"""


import serial
import time

class velmex_vxm:
	
	"""
	
	#TODO add content
	
	Notes
	-----
	* manual : https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
	* Based on code from here : https://github.com/unkatoco/Velmex/blob/6443294f597839bdb9d9b18d2b47abb6c4e745e8/velmex.py
	* more code : https://github.com/mattgbiz/Velmex-XYZ-StageControl-Python/blob/master/XYZMotion.py
	"""

	
    
	# returns the position of motor m    
	def get_motor_position (self, motor_num=1):
		motors = ['X','Y','Z','T' ]
		global ser
		self.instrument.readline();          # clear current buffer
		self.instrument.write(motors[motor_num-1].encode())   # query for position
		pos = int(self.instrument.readline()[:-1])
		
		return pos*self.inches_per_step
	
	def check_status(self, verbose=True):
		self.instrument.readline()                  # clear current buffer
		self.instrument.write(b"V")					# query for status
		status = self.instrument.readline()			# read status
		if verbose==True:
			if status == b"R":
				print('Unit is ready')
			elif status == b"B":
				print('Unit is busy')
			elif status == b"J":
				print('Unit is in Jog/slew mode')
			elif status == b"b":
				print('Unit is Jog/slewing')
			else:
				print('Status not recognized:',status)
		return status
		
	
	def set_motor_speed(self,  speed=2000, motor_num=1,):
		
		self.instrument.write(b"C")  # clear current program
		command = "S" + str(motor_num) + "M" + str(speed) + ","
		self.instrument.write(command.encode()) # send movement command
		self.instrument.write(b"R")  # run current program
	
	def __init__(	self, 
					port = 'COM6', 
					baudrate=9600, 
					bytesize=8,  
					parity=serial.PARITY_NONE, 
					stopbits=1, 
					timeout=.1, 
					steps_per_revolution = 360/0.9, 
					inches_per_revolution = 0.1):
		
		self.instrument= serial.Serial(	port=port, 
										baudrate=baudrate, 
										bytesize=bytesize , 
										parity=parity, 
										stopbits=stopbits, 
										timeout=timeout)
		self.instrument.write(b"F")  # enable online mode (with echo off)
		self.instrument.write(b"C")  # clear current program
		if self.check_status(verbose=False) == b'R':
			print('Successfully connected at port:', port)
			
		self.steps_per_revolution = steps_per_revolution
		self.inches_per_revolution = inches_per_revolution
		self.inches_per_step = inches_per_revolution / steps_per_revolution
		self.steps_per_inch = steps_per_revolution / inches_per_revolution 
		
	def _move_motor(self, command, wait=True):
		self.instrument.write(b"C")  # clear current program
		self.instrument.write(command.encode()) # send movement command
		self.instrument.write(b"R")  # run current program
		if wait == True:
			self.wait_until_done()
		
	def move_by (self, distance, motor_num=1, wait=True):
	    self._move_motor("I" + str(motor_num) + "M" + str(int(distance*self.steps_per_inch)) + ",", wait=wait) # send movement command
	
	def move_to (self, destination, motor_num=1, wait=True):
	    self._move_motor("IA" + str(motor_num) + "M" + str(int(destination*self.steps_per_inch)) + ",", wait=wait) # send movement command
		
	def home_motor(self, motor_num, wait=True):
	    self._move_motor("IA" + str(motor_num) + "M0" + ",", wait=wait) # send movement command
		
	def wait_until_done(self):
		""" delay until current program is done    """
		
		self.instrument.readline()                  # clear current buffer
		self.instrument.write(b'V')                  # query for velmex's status 
		while (self.instrument.readline()==b"B"):    # if busy, loop until not busy
			time.sleep(0.01)
			self.instrument.write(b"V")      
			
	def disconnect(self):
		self.instrument.close()
		
	def find_limits(self, motor_num=1, verbose=True):
		self.move_to(1000, motor_num=motor_num)
		lim_max=self.get_motor_position(motor_num=motor_num)
		self.move_to(-1000, motor_num=motor_num)
		lim_min=self.get_motor_position(motor_num=motor_num)
		
		if verbose==True:
			print('limits: ',lim_max,lim_min)
			print('total range of motion',lim_max-lim_min)
		
		return lim_min,lim_max
	
	def set_home(self):
		self.instrument.write(b'N') 

if __name__=="__main__":
	unit=velmex_vxm()
# 	limits=unit.find_limits()
# 	print(limits)