# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:01:01 2021

@author: jwbrooks
"""

import johnspythonlibrary2 as jpl2
import nrl_code as nrl
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# from johnspythonlibrary2.Instruments.velmex_vxm import velmex_vxm

# motor=velmex_vxm('COM8')



import time
import serial



class close_the_loop:
 	
	def __init__(self, vxm, vro):
		
		self.vxm=vxm
		self.vro=vro
		
	def move_to(self, position, motor_num, delta=0.01):
		
		time.sleep(0.2)
		
		check = True
		count=0
		while( check == True ):
			
			vro_position=self.vro.get_motor_position(motor_num=motor_num)
			difference = vro_position - position
			print(vro_position, position, difference)
				
			if np.abs(difference)<delta:
				check = False
			else:
				self.vxm.move_by(-difference, motor_num=motor_num)
			count+=1
			if count>10:
				print('gave up after 10 attempts')
				check=False
				

class velmex_vxm:
	
	"""
	
	#TODO add content
	
	Notes
	-----
	* manual : https://www.velmex.com/Downloads/User_Manuals/vxm_user_manl.pdf
	* Based on code from here : https://github.com/unkatoco/Velmex/blob/6443294f597839bdb9d9b18d2b47abb6c4e745e8/velmex.py
	* more code : https://github.com/mattgbiz/Velmex-XYZ-StageControl-Python/blob/master/XYZMotion.py
	"""

	
	import serial
    
	# returns the position of motor m    
	def get_motor_position (self, motor_num=1):
		
		if motor_num==1:
			steps_per_inch=self.m1_settings['steps_per_inch']
		elif motor_num==2:
			steps_per_inch=self.m2_settings['steps_per_inch']
			
		motors = ['X','Y','Z','T' ]
# 		global ser
		self.instrument.read_all();          # clear current buffer
		time.sleep(0.1)
		self.instrument.write(motors[motor_num-1].encode()) 
		time.sleep(0.1)
		
		if True:
			if motor_num==1:
				steps_per_inch=self.m1_settings['steps_per_inch']
			elif motor_num==2:
				steps_per_inch=self.m2_settings['steps_per_inch']
			
			return int(self.instrument.read_all().decode())/steps_per_inch
		if False:
			position=int(self.instrument.read_all().decode())  # query for position
	
			
			if motor_num==1:
				steps_per_inch=self.m1_settings['steps_per_inch']
			elif motor_num==2:
				steps_per_inch=self.m2_settings['steps_per_inch']
			
			position/=self.m1_settings['steps_per_inch']
			
			return position
	
	
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
	
	def write_command(self, cmd): #TODO confirm this command works
		# make sure cmd is binary
		if type(cmd) is str:
			cmd=cmd.encode()
		# make sure cmd ends in a carriage return
		if cmd[-1]!=b'\r':
			cmd+=b'\r'
		# write command
		self.instrument.write(cmd)
		
	def query(self, cmd, verbose=False): #TODO confirm this command works
	
		# clear buffer
		self.instrument.read_all()
		
		# write command
		self.write_command(cmd)
		
		# get response
		time.sleep(0.1)
		response = self.instrument.read_all()
		
		if verbose==True:
			print(response)
		
		return response
	
	def set_motor_speed(self,  speed=200, motor_num=1,):
		
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
					m1_steps_per_revolution = 180/0.9, 
					m2_steps_per_revolution = 360/0.9,
					m1_inches_per_revolution = 0.05,
					m2_inches_per_revolution = 0.1):
		
		self.instrument= serial.Serial(	port=port, 
										baudrate=baudrate, 
										bytesize=bytesize, 
										parity=parity, 
										stopbits=stopbits, 
										timeout=timeout)
		self.instrument.write(b"F")  # enable online mode (with echo off)
		self.instrument.write(b"C")  # clear current program
		if self.check_status(verbose=False) == b'R':
			print('Successfully connected at port:', port)
			
		time.sleep(0.1)
		self.instrument.read_all()
		self.instrument.write(b'getD3')
		self.instrument.write(b'getD0')
		self.instrument.write(b'getD1')
		time.sleep(0.1)
		print('Connected to', self.instrument.read_all())
			
		self.m1_settings={'steps_per_revolution':m1_steps_per_revolution, 'inches_per_revolution':m1_inches_per_revolution, 'inches_per_step':m1_inches_per_revolution / m1_steps_per_revolution, 'steps_per_inch' : m1_steps_per_revolution / m1_inches_per_revolution }
		self.m2_settings={'steps_per_revolution':m2_steps_per_revolution, 'inches_per_revolution':m2_inches_per_revolution, 'inches_per_step':m2_inches_per_revolution / m2_steps_per_revolution, 'steps_per_inch' : m2_steps_per_revolution / m2_inches_per_revolution }
		
		self.set_motor_speed(500,1)
		self.set_motor_speed(500,2)
# 		self.steps_per_revolution = steps_per_revolution
# 		self.inches_per_revolution = inches_per_revolution
# 		self.inches_per_step = inches_per_revolution / steps_per_revolution
# 		self.steps_per_inch = steps_per_revolution / inches_per_revolution 
		
	def _move_motor(self, command, wait=True):
		print(command)
		self.instrument.write(b"C")  # clear current program
		self.instrument.write(command.encode()) # send movement command
		self.instrument.write(b"R")  # run current program
		if wait == True:
			self.wait_until_done()
		
	def move_by (self, distance, motor_num=1, wait=True):
		if motor_num==1:
			steps_per_inch=self.m1_settings['steps_per_inch']
		elif motor_num==2:
			steps_per_inch=self.m2_settings['steps_per_inch']
		self._move_motor("I" + str(motor_num) + "M" + str(int(distance*steps_per_inch)) + ",", wait=wait) # send movement command
	
	def move_to (self, destination, motor_num=1, wait=True):
		if motor_num==1:
			steps_per_inch=self.m1_settings['steps_per_inch']
		elif motor_num==2:
			steps_per_inch=self.m2_settings['steps_per_inch']
		self._move_motor("IA" + str(motor_num) + "M" + str(int(destination*steps_per_inch)) + "\r", wait=wait) # send movement command
		
	def home_motor(self, motor_num, wait=True):
	    self._move_motor("IA" + str(motor_num) + "M0" + ",", wait=wait) # send movement command
		
	def wait_until_done(self, sleep_time=0.01):
		""" delay until current program is done    """
		
		self.instrument.readline()                  # clear current buffer
		self.instrument.write(b'V')                  # query for velmex's status 
		while (self.instrument.readline()==b"B"):    # if busy, loop until not busy
			time.sleep(sleep_time)
			self.instrument.write(b"V")      
			
	def disconnect(self):
		self.instrument.write(b'Q')
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
		
	def set_position_as(self, motor_num=1, pos=0): #TODO confirm this command works
		if motor_num==1:
			steps_per_inch=self.m1_settings['steps_per_inch']
		elif motor_num==2:
			steps_per_inch=self.m2_settings['steps_per_inch']
# 		self.instrument.write(b'IA%dM%d'%(motor_num,int(pos))) 
		command='E,C,IA1M%d,R\r'%int(steps_per_inch*pos)
		self.instrument.write(command.encode())
		
		#  clear buffer
		self.instrument.read_all()

		
# 	def close_the_loop(self, encoder, motor_num=1): #TODO write this command
# 		print('work in progress') 
# 		encoder_position = encoder.get_motor_position(motor_num=motor_num)
# # 		if encoder_units == 'in' and motor_num==1:
# # 			encoder_position*=self.m1_settings['steps_per_inch'] 
# # 		elif encoder_units == 'in' and motor_num==2:
# # 			encoder_position*=self.m2_settings['steps_per_inch'] 
 			
# 		controller_position=self.get_motor_position(motor_num=motor_num)
# 		
# 		difference = encoder_position - controller_position
# 		self.move_


class velmex_vro:
	
	"""
	
	#TODO add content
	
	Notes
	-----
	# # https://www.velmex.com/Downloads/User_Manuals/VRO%20Reference%20Manual.pdf

	"""

	
	import serial
    
	# returns the position of motor m    
	def get_motor_position (self, motor_num=1):
		
# 		if motor_num==1:
# 			steps_per_inch=self.m1_settings['steps_per_inch']
# 		elif motor_num==2:
# 			steps_per_inch=self.m2_settings['steps_per_inch']
# 			
			
		motors = ['x','y','z','t' ]
		global ser
		self.instrument.read_all();          # clear current buffer
		self.instrument.write(motors[motor_num-1].encode())   # query for position
		pos = float(self.instrument.readline()[:-1])
		
		return pos#*steps_per_inch
	
	
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
		
	
# 	def set_motor_speed(self,  speed=500, motor_num=1,):
# 		
# 		self.instrument.write(b"C")  # clear current program
# 		command = "S" + str(motor_num) + "M" + str(speed) + ","
# 		self.instrument.write(command.encode()) # send movement command
# 		self.instrument.write(b"R")  # run current program
	
	def __init__(	self, 
					port = 'COM6', 
					baudrate=9600, 
					bytesize=8,  
					parity=serial.PARITY_NONE, 
					stopbits=1, 
					timeout=.1, 
					m1_multiplier=int(1),
					m1_divisor=int(4000),#180/1.8/0.1
					m2_multiplier=int(1),
					m2_divisor=int(360/1.8/0.1),
# 					m1_steps_per_revolution = 180/0.9, 
# 					m2_steps_per_revolution = 360/0.9,
# 					m1_inches_per_revolution = 0.1,
# 					m2_inches_per_revolution = 0.1
# 					steps_per_revolution = 360/0.9, 
# 					inches_per_revolution = 0.1,
					):
		
		self.instrument= serial.Serial(	port=port, 
										baudrate=baudrate, 
										bytesize=bytesize , 
										parity=parity, 
										stopbits=stopbits, 
										timeout=timeout)
		self.instrument.write(b"F")  # enable online mode (with echo off)
		# self.instrument.write(b"C")  # clear current program (Note that this command zeros the )
		if self.check_status(verbose=False) == b'R':
			print('Successfully connected at port:', port)
	
		## Print details about the unit
		self.instrument.write(b'getD3')
		self.instrument.write(b'getD0')
		self.instrument.write(b'getD1')
		time.sleep(0.1)
		print('Connected to', self.instrument.read_all())
		
		## Set default encoder constants
		
		self.write('set*X%d'%int(1*127))
		self.write('set/X%d'%int(1*1000))
		self.write('set*Y%d'%int(1*127))
		self.write('set/Y%d'%int(1*100))
# 		self.write('set*Y%d'%int(1*254))
# 		self.write('set/Y%d'%int(1*1000))
# 		self.write('set*X%d'%int(1*127))
# 		self.write('set/X%d'%int(1*100))
		
		self.write('set*x%d'%int(50))
		self.write('set/x%d'%int(1))
		self.write('set*y%d'%int(50))
		self.write('set/y%d'%int(1))
		
		self.instrument.write(b"Q")  # Retrun to standard dispaly mode
		
# 		self.instrument.write(b'setUXin')
# 		self.instrument.write(b'setUYin')

# 		self.m1_settings={'steps_per_revolution':m1_steps_per_revolution, 'inches_per_revolution':m1_inches_per_revolution, 'inches_per_step':m1_inches_per_revolution / m1_steps_per_revolution, 'steps_per_inch' : m1_steps_per_revolution / m1_inches_per_revolution }
# 		self.m2_settings={'steps_per_revolution':m2_steps_per_revolution, 'inches_per_revolution':m2_inches_per_revolution, 'inches_per_step':m2_inches_per_revolution / m2_steps_per_revolution, 'steps_per_inch' : m2_steps_per_revolution / m2_inches_per_revolution }
		
			
# 		self.steps_per_revolution = steps_per_revolution
# 		self.inches_per_revolution = inches_per_revolution
# 		self.inches_per_step = inches_per_revolution / steps_per_revolution
# 		self.steps_per_inch = steps_per_revolution / inches_per_revolution 
		
	def write(self, cmd):
		if type(cmd)==str:
			cmd=cmd.encode()
		self.instrument.write(cmd+'\r'.encode())
		
		
	def write_and_read(self, cmd, verbose=False):
		self.write(cmd)
		time.sleep(0.1)
		ret = self.instrument.read_all()
		if verbose == True:
			print(ret)
			
		return ret
		
	def zero_positions(self):
		self.instrument.write(b"C")
		
	def _move_motor(self, command, wait=True):
		self.instrument.write(b"C")  # clear current program
		self.instrument.write(command.encode()) # send movement command
		self.instrument.write(b"R")  # run current program
		if wait == True:
			self.wait_until_done()
# 		
# 	def move_by (self, distance, motor_num=1, wait=True):
# 	    self._move_motor("I" + str(motor_num) + "M" + str(int(distance*self.steps_per_inch)) + ",", wait=wait) # send movement command
# 	
# 	def move_to (self, destination, motor_num=1, wait=True):
# 	    self._move_motor("IA" + str(motor_num) + "M" + str(int(destination*self.steps_per_inch)) + ",", wait=wait) # send movement command
# 		
# 	def home_motor(self, motor_num, wait=True):
# 	    self._move_motor("IA" + str(motor_num) + "M0" + ",", wait=wait) # send movement command
# 		
# 	def wait_until_done(self, sleep_time=0.01):
# 		""" delay until current program is done    """
# 		
# 		self.instrument.readline()                  # clear current buffer
# 		self.instrument.write(b'V')                  # query for velmex's status 
# 		while (self.instrument.readline()==b"B"):    # if busy, loop until not busy
# 			time.sleep(sleep_time)
			self.instrument.write(b"V")      
			
	def disconnect(self):
		self.instrument.write(b'Q')
		self.instrument.close()
		
# 	def find_limits(self, motor_num=1, verbose=True):
# 		self.move_to(1000, motor_num=motor_num)
# 		lim_max=self.get_motor_position(motor_num=motor_num)
# 		self.move_to(-1000, motor_num=motor_num)
# 		lim_min=self.get_motor_position(motor_num=motor_num)
# 		
# 		if verbose==True:
# 			print('limits: ',lim_max,lim_min)
# 			print('total range of motion',lim_max-lim_min)
# 		
# 		return lim_min,lim_max
	
# 	def set_home(self):
# 		self.instrument.write(b'N') 

if __name__=="__main__":
	vxm=velmex_vxm('COM8')
# 	limits=unit.find_limits()
# 	print(limits)