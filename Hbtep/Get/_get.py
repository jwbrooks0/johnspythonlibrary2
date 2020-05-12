
#%% Libraries

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import os.path as _path

from johnspythonlibrary2.Hbtep.Process import leastSquareModeAnalysis as _leastSquareModeAnalysis
from johnspythonlibrary2.ReadWrite import readOdsToDF as _readOdsToDF
from johnspythonlibrary2.ReadWrite import backupDFs as _backupDFs
from johnspythonlibrary2.Process.Filters import gaussianFilter_df as _gaussianFilter_df
import johnspythonlibrary2.Plot._plot as _plot
import johnspythonlibrary2.Hbtep.Plot._plot as _plotHbt
from johnspythonlibrary2.Process.Pandas import filterDFByTime as _filterDFByTime
from johnspythonlibrary2.Process.Pandas import filterDFByColOrIndex as _filterDFByColOrIndex


import MDSplus as _mds
 

#%% Settings
   
# This file does not exist by default.  Rename the template file and fill in the appropriate variables within.
import johnspythonlibrary2.Hbtep.Get._settings as _settings

_TSTART=0e-3
_TSTOP=10e-3


#%% Functions - MDSplus related

def latestShotNumber(serverAddress=_settings._SERVER_ADDRESS,
					 port=_settings._SERVER_PORT):
	"""
	Gets the latest shot number from the tree
	
	Parameters
	----------
	
	Returns
	-------
	shot_num : int
		latest shot number
	"""
	
	conn = _mds.Connection('%s:%d'%(serverAddress,port));
	shot_num = conn.get('current_shot("hbtep2")')
	return int(shot_num)

		
def mdsData(shotno,
			dataAddress=['\HBTEP2::TOP.DEVICES.SOUTH_RACK:CPCI_10:INPUT_94',
						 '\HBTEP2::TOP.DEVICES.SOUTH_RACK:CPCI_10:INPUT_95'],
			tStart=0e-3,
			tStop=10e-3,
			columnNames=[]):
	"""
	Get data and optionally associated time from MDSplus tree
	
	Parameters
	----------
	shotno : int
		shotno of data.  
	dataAddress : list (of strings)
		address of desired data on MDSplus tree
	tStart : float
		trims data before this time
	tStop : float
		trims data after this time
	
	Returns
	-------
	df : dataframe
		A pandas dataframe of the data
		
	Examples
	--------
	df=mdsData(100000)
	"""			
	
	## libraries
	import MDSplus as _mds
	
	## subfunctions
	def _initRemoteMDSConnection(	shotno,
									serverAddress=_settings._SERVER_ADDRESS,
									port=_settings._SERVER_PORT):
		"""
		Initiate remote connection with MDSplus HBT-EP tree
		
		Parameters
		----------
		shotno : int
			The shotnumber to store the data
		serverAddress : str
			The network/internet address of the server
		port : int
			The port of the server
		
		Returns
		-------
		conn : MDSplus.connection
			connection class to mdsplus tree on the server for the specific shotno
		"""
		conn = _mds.Connection('%s:%d'%(serverAddress,port));
		conn.openTree('hbtep2', int(shotno));
		return conn
	
	def _closeRemoteMDSConnection(conn,shotno):
		"""
		Closes remote connection with MDSplus HBT-EP tree
		
		Parameters
		----------
		conn : MDSplus.connection
			connection class to mdsplus tree on the server for the specific shotno
		shotno : int
			The shotnumber of the data
		
		"""
		mdsConn.closeTree('hbtep2', shotno)
		mdsConn.disconnect()
		
	## make sure inputs are correct 
	if type(dataAddress) is not list:
		dataAddress=[dataAddress];
		
	shotno=int(shotno)
	
	mdsConn=_initRemoteMDSConnection(shotno);
	
		
	## create a dataframe of data for each provided shotnumber
	df=_pd.DataFrame()
	for j,jkey in enumerate(dataAddress):
		data=mdsConn.get(jkey).data()
		time=mdsConn.get('dim_of('+jkey+')').data()
		
		# sometimes data and time don't have the same dimensions.  I don't know why.
		if time.shape[0]!=data.shape[0]:
			Warning('Time and data dimensions do not agree.  %d and %d, respetively.'%(time.shape[0],data.shape[0]))
			
			# when this happens, typically time is only 1 size larger.  this fixes this one case.  shot 70246 is an example
			if time.shape[0]-1 == data.shape[0]:
				time=time[:-1]
				
		ds=_pd.Series(data,
						time,
						name=jkey)
		
		ds=ds[(ds.index>=tStart)&(ds.index<=tStop)]
		df[jkey]=ds
		
	if columnNames!=[]:
		df.columns=columnNames
	
	_closeRemoteMDSConnection(mdsConn,shotno)
		
	return df


#%% Functions - HBT data related


def capBankData(	shotno=96530,
					tStart=_TSTART,
					tStop=_TSTOP,
					plot=False):
	"""
	Capacitor bank data.  Currents.  
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Example
	-------
	::
		
		capBankData(96530,plot=True)
		
	"""
		
	def plotData(df,dfTF):
		""" Plot all relevant plots """
		
		_plt.figure()
		ax1 = _plt.subplot2grid((3,2), (0,1), rowspan=3)  #tf
		ax2 = _plt.subplot2grid((3,2), (0,0)) #vf
		ax3 = _plt.subplot2grid((3,2), (1,0),sharex=ax2) #oh
		ax4 = _plt.subplot2grid((3,2), (2, 0),sharex=ax2) #sh
		fig=_plt.gcf()
		fig.set_size_inches(10,5)
				
		ax1.plot(dfTF.index,dfTF.TF)
		ax1.axvspan(df.index[0],df.index[-1],color='r',alpha=0.3)
		_plot.finalizeSubplot(	ax1,
								xlabel='Time (s)',
#								xlim=[-150,450],
								ylabel='TF Field (T)',
								title='%d'%shotno)
		
		ax2.plot(df.index*1e3,df.VF_CURRENT*1e-3)
		_plot.finalizeSubplot(	ax2,
								ylabel='VF Current\n(kA)')
		
		ax3.plot(df.index*1e3,df.OH_CURRENT*1e-3)
		_plot.finalizeSubplot(	ax3,
								ylim=[-20,30],
								ylabel='OH Current\n(kA)')
		
		ax4.plot(df.index*1e3,df.SH_CURRENT*1e-3)
		_plot.finalizeSubplot(	ax4,
								ylim=[tStart,tStop],
								xlabel='Time (s)',
								ylabel='SH Current\n(kA)')
		
	# get vf data
	
	@_backupDFs
	def capBankData(shotno,
					 tStart,
					 tStop):
		df=mdsData(	shotno=shotno,
					  dataAddress=['\HBTEP2::TOP.SENSORS.VF_CURRENT',
								  '\HBTEP2::TOP.SENSORS.OH_CURRENT',
								  '\HBTEP2::TOP.SENSORS.SH_CURRENT'],
					  tStart=tStart, 
					  tStop=tStop,
					  columnNames=['VF_CURRENT','OH_CURRENT','SH_CURRENT']) 
		return df
	
	df=capBankData(shotno,tStart=tStart,tStop=tStop)
	df=_filterDFByTime(df,tStart,tStop)

	if plot == True:	
		# get TF data
		dfTF=tfData(shotno,
				tStart=-0.25,
				tStop=0.5)
		
		# plot
		plotData(df,dfTF)
		
	return df


def cos1RogowskiData(	shotno=96530,
						tStart=_TSTART,
						tStop=_TSTOP,
						plot=False):
	"""
	Gets cos 1 rogowski data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Example
	-------
	::
		
		df=cos1RogowskiData(96530,plot=True)
		
	"""
	# get data.  need early time data for offset subtraction
	
	@_backupDFs
	def getCos1RogData(shotno,
					tStart,
					tStop):
		dfData=mdsData(shotno=shotno,
							  dataAddress=['\HBTEP2::TOP.SENSORS.ROGOWSKIS:COS_1',
										   '\HBTEP2::TOP.SENSORS.ROGOWSKIS:COS_1:RAW'],
							  tStart=-1e-3, 
							  tStop=tStop,
							  columnNames=['COS_1','COS_1_RAW'])
		return dfData
	
	dfData=getCos1RogData(shotno,
					   tStart=tStart,
					   tStop=tStop)

	# remove offest
	dfData['COS_1_RAW']-=dfData.COS_1_RAW[dfData.COS_1_RAW.index<0].mean()
	
	# trim time
	dfData=dfData[(dfData.index>=tStart)&(dfData.index<=tStop)]

	if plot==True:
		dfData.plot()
	
	return dfData
		

def egunData(	shotno=96530,
				tStart=_TSTART,
				tStop=_TSTOP,
				plot=False):
	"""
	Gets egun data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Returns
	-------
	dfData : pandas.core.frame.DataFrame
		e-gun data
		
	Example
	-------
	::
		
		df=egunData(100000,plot=True)
	
	"""
	
	# get data
	dfData=mdsData(	shotno,
					dataAddress=['\HBTEP2::TOP.OPER_DIAGS.E_GUN:I_EMIS', 
								'\HBTEP2::TOP.OPER_DIAGS.E_GUN:I_HEAT',
								'\HBTEP2::TOP.OPER_DIAGS.E_GUN:V_BIAS',],
					tStart=tStart,
					tStop=tStop,
					columnNames=['I_EMIS','I_HEAT','V_BIAS'])

	# calculate RMS heating current
	dfData['I_HEAT_RMS']=_np.sqrt(_np.average((dfData.I_HEAT-_np.average(dfData.I_HEAT))**2));

	# optional plot
	if plot==True:
		dfData.plot()
		
	return dfData



def euvData(	shotno=101393,
				tStart=_TSTART,
				tStop=_TSTOP,
				plot=False):
	"""""
	Get EUV fan array data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		default is False
		True - plots far array of all 11 (of 16) channels
		'all' - plots 
		
		
	Example
	-------
	::
		
		df=euvData(101393,plot=True)
	"""""
	
	
	# subfunctions
	@_backupDFs
	def dfEUV(shotno, tStart, tStop,	dfEUVMeta):
		dfEUV=mdsData(shotno,dfEUVMeta.addresses.to_list(), tStart, tStop,	columnNames=dfEUVMeta.index.to_list())
		return dfEUV
	
	# load meta data
	sensor='EUV'
	try:
		directory=_path.dirname(_path.realpath(__file__))
		dfMeta=_readOdsToDF('%s/listOfAllSensorsOnHBTEP.ods'%directory,sensor).set_index('names')
	except:
		dfMeta=_readOdsToDF('listOfAllSensorsOnHBTEP.ods',sensor).set_index('names')
	
	# load raw data
	df=dfEUV(shotno,tStart,tStop,dfMeta)
	df=_filterDFByTime(df,tStart,tStop)
	
	if plot==True:
		df.plot()
		
	return df


def ipData(	shotno=96530,
			tStart=_TSTART,
			tStop=_TSTOP,
			plot=False,
			findDisruption=True,
			verbose=False,
			paIntegrate=True,
			forceDownload=False):
	"""
	Gets plasma current (I_p) data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	findDisruption : bool
		Optional.  Finds the time of disruption.
	verbose : bool
		Plots intermediate steps associated with the time of disruption calculation
	paIntegrate : bool
		Integrates the PA1 and PA2 sensors to provide an alternative Ip measurement
		
	Returns
	-------
	dfIp : pandas.core.frame.DataFrame
		Ip current from the Ip Rogowski
		(Optional) Ip current from the PA1 and PA2 sensors
		(Optional) Time of disruption
	
	Example:
	--------
	::
		
		ipData(96530,plot=True)
	
	"""
	
	# subfunctions
	@_backupDFs
	def dfIpData(shotno, tStart, tStop):
		dataAddress=['\HBTEP2::TOP.SENSORS.ROGOWSKIS:IP']
		df=mdsData(shotno,dataAddress, tStart, tStop,	columnNames=['IpRog'])
		return df
	
	# download data form IPRogowski data
	dfIp=dfIpData(	shotno,
					tStart=tStart, 
					tStop=tStop,
					forceDownload=forceDownload)
	dfIp=_filterDFByTime(dfIp,tStart,tStop)
	
	# integrate PA1 sensor data to get IP
	if paIntegrate==True:
		# constants
		mu0=4*_np.pi*1e-7
		minorRadius=0.16
		
		for key in ['PA1','PA2']:
			dfPA,_,_=magneticSensorData(shotno,tStart,tStop,sensor=key,forceDownload=forceDownload)
			ipPAIntegration=_np.array(dfPA.sum(axis=1)*1.0/dfPA.shape[1]*2*_np.pi*minorRadius/mu0)
			dfIp['ip%s'%key]=ipPAIntegration

	# finds the time of disruption
	if findDisruption==True:
		
		try:
			dfTemp=_pd.DataFrame(dfIp.IpRog[dfIp.index>1.5e-3])
			
			# filter data 
			dfTemp['HP']=_gaussianFilter_df(dfTemp,
							  timeFWHM=0.5e-3,
							  filterType='high',
							  plot=False)
			dfTemp['LP']=_gaussianFilter_df(_pd.DataFrame(dfTemp['HP']),
							  timeFWHM=0.01e-3,
							  filterType='low',
							  plot=False)
			
			# find time derivative of smoothed ip
			dfTemp['dip2dt']=_np.gradient(dfTemp.LP.to_numpy())				

			# find the first large rise in d(ip2)/dt
			threshold=11.0
			index=_np.where(dfTemp.dip2dt>threshold)[0][0]
			
			# find the max value of ip immediately after the disrup. onset
			while(dfTemp.IpRog.to_numpy()[index]<dfTemp.IpRog.to_numpy()[index+1]):
				index+=1
			tDisrupt=dfTemp.iloc[index].name
			
			if verbose:			
				dfTemp['timeDisruption']=_np.zeros(dfTemp.shape[0])
				dfTemp['timeDisruption'].at[tDisrupt]=dfTemp.IpRog.at[tDisrupt]
				dfTemp.plot()

			dfIp['timeDisruption']=_np.zeros(dfIp.shape[0])
			dfIp['timeDisruption'].at[tDisrupt]=dfIp.IpRog.at[tDisrupt]

			
		except:
			print("time of disruption could not be found")
	
	if plot == True:
		fig, ax=_plt.subplots()
		for i,(key,val) in enumerate(dfIp.iteritems()):
			ax.plot(val.index*1e3,val,label=key)
		_plot.finalizeSubplot(	ax,
								xlabel='Time (ms)',
								ylabel='Current (A)',
								title='%d'%shotno)
		
	return dfIp

def loopVoltageData(	shotno=96530,
						tStart=_TSTART,
						tStop=_TSTOP,
						plot=False):
	"""
	lopo voltage data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Returns
	-------
	dfData : pandas.core.frame.DataFrame
		Loop voltage data.  Time is index.
		
	Example
	-------
	::
		
		df=loopVoltageData(100000,plot=True)
	
	"""

	# get data
	dfData=mdsData(shotno=shotno,
						  dataAddress=['\HBTEP2::TOP.SENSORS.LOOP_VOlTAGE'],
						  tStart=tStart, tStop=tStop)   

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(dfData.index*1e3,dfData)
		_plot.finalizeSubplot(	ax,
								xlabel='Time',
								ylabel='V',
								title='%d'%shotno,
								ylim=[3,15],
								legendOn=False)
		
	return dfData


def magneticSensorDataAll(	shotno,
							tStart=_TSTART,
							tStop=_TSTOP,
							plot=False,
							removeBadSensors=True,
							timeFWHMSmoothing=0.4e-3,
							forceDownload=False):
	"""
	
	Downloads magnetic sensor data for all sensors.  Presently, only poloidal
	measurements as the radial sensors are not yet implemeted.  
	
	Parameters
	----------

	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	removeBadSensors : bool
		removes known bad sensors
	timeFWHMSmoothing : float
		Time constant associated with the offset subtraction filter
		
	Returns
	-------
	dfRawAll : pandas.core.frame.DataFrame
		Raw magnetic data.  Time is index.
	dfSmoothedAll : pandas.core.frame.DataFrame
		Offset subtracted magnetic data.  Time is index.
	dfMetaAll : pandas.core.frame.DataFrame
		Meta data for the sensors
		
	Examples
	--------
	Example1::
		
		dfRaw,dfSmoothed,dfMeta=magneticSensorDataAll(106000,plot=True,tStart=1.5e-3,tStop=5.5e-3)
		
	"""
	
	sensors=['TA','PA1','PA2','FB']
	for sensor in sensors:
		dfRaw,dfSmoothed,dfMeta=magneticSensorData(	shotno,
													tStart=tStart,
													tStop=tStop,
													plot=plot,
													removeBadSensors=removeBadSensors,
													sensor=sensor,
													forceDownload=forceDownload)

		if sensor=='TA':
			dfRawAll=dfRaw.copy()
			dfSmoothedAll=dfSmoothed.copy()
			dfMetaAll=dfMeta.copy()
			
		else:
			dfRawAll=_pd.concat((dfRawAll,dfRaw),axis=1)
			dfSmoothedAll=_pd.concat((dfSmoothedAll,dfSmoothed),axis=1)
			dfMetaAll=_pd.concat((dfMetaAll,dfMeta))
		
	return dfRawAll, dfSmoothedAll, dfMetaAll
		

def magneticSensorData(	shotno=98173,
						tStart=_TSTART,
						tStop=_TSTOP,
						plot=False,
						removeBadSensors=True,
						sensor='TA',
						timeFWHMSmoothing=0.4e-3,
						forceDownload=False):
	"""
	Downloads magnetic sensor data.  Presently, only poloidal
	measurements as the radial sensors are not yet implemeted.  
	
	Parameters
	----------

	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	removeBadSensors : bool
		removes known bad sensors
	sensor : str
		sensor to evalualte.  Must be in ['TA','PA1','PA2','FB']
	timeFWHMSmoothing : float
		Time constant associated with the offset subtraction filter
		
	Returns
	-------
	dfRaw : pandas.core.frame.DataFrame
		Raw magnetic data.  Time is index.
	dfSmoothed : pandas.core.frame.DataFrame
		Offset subtracted magnetic data.  Time is index.
	dfMeta : pandas.core.frame.DataFrame
		Meta data for the sensors
	
	Notes
	-----
	other possible bad sensors: ['TA02_S1P','TA07_S3P','TA10_S3P'] 
	The sensors that are bad are not necessarily consistent from shot to shot or year to year
 
	
	Examples
	--------
	::
		
		shotno=106000
		a,b,c=magneticSensorData(shotno,sensor='TA',plot=True,tStart=1.5e-3,tStop=5.5e-3)
		a,b,c=magneticSensorData(shotno,sensor='PA1',plot=True,tStart=1.5e-3,tStop=5.5e-3)
		a,b,c=magneticSensorData(shotno,sensor='PA2',plot=True,tStart=1.5e-3,tStop=5.5e-3)
		a,b,c=magneticSensorData(shotno,sensor='FB',plot=True,tStart=1.5e-3,tStop=5.5e-3,forceDownload=True)

	"""
	
	if sensor not in ['TA','PA1','PA2','FB']:
		raise Exception('Bad sensor name')
	
	# subfunctions
	@_backupDFs
	def dfTARaw(shotno, dfTAMeta, tStart=_TSTART, tStop=_TSTOP,	):
		dfTARaw=mdsData(shotno,dfTAMeta.addresses.to_list(), tStart, tStop,	columnNames=dfTAMeta.index.to_list())
		return dfTARaw
	
	@_backupDFs
	def dfPA1Raw(shotno, dfPA1Meta, tStart=_TSTART, tStop=_TSTOP):
		dfPA1Raw=mdsData(shotno,dfPA1Meta.addresses.to_list(), tStart, tStop,	columnNames=dfPA1Meta.index.to_list())
		return dfPA1Raw
	
	@_backupDFs
	def dfPA2Raw(shotno, dfPA2Meta, tStart=_TSTART, tStop=_TSTOP):
		dfPA2Raw=mdsData(shotno,dfPA2Meta.addresses.to_list(), tStart, tStop,	columnNames=dfPA2Meta.index.to_list())
		return dfPA2Raw
	
	@_backupDFs
	def dfFBRaw(shotno, dfFBMeta, tStart=_TSTART, tStop=_TSTOP):
		dfFBRaw=mdsData(shotno,dfFBMeta.addresses.to_list(), tStart, tStop,	columnNames=dfFBMeta.index.to_list())
		return dfFBRaw
	
	# load meta data
	try:
		directory=_path.dirname(_path.realpath(__file__))
		dfMeta=_readOdsToDF('%s/listOfAllSensorsOnHBTEP.ods'%directory,sensor).set_index('names')
	except:
		dfMeta=_readOdsToDF('listOfAllSensorsOnHBTEP.ods',sensor).set_index('names')
		
	# load raw data
	if sensor=='TA':
		dfRaw=dfTARaw(shotno,dfMeta,forceDownload=forceDownload)
	elif sensor=='PA1':
		dfRaw=dfPA1Raw(shotno,dfMeta,forceDownload=forceDownload)
	elif sensor=='PA2':
		dfRaw=dfPA2Raw(shotno,dfMeta,forceDownload=forceDownload)
	elif sensor=='FB':
		dfRaw=dfFBRaw(shotno,dfMeta,forceDownload=forceDownload)
	if removeBadSensors:
		dfRaw=dfRaw.drop(columns=dfMeta[dfMeta.bad==True].index.to_list())
		dfMeta=dfMeta.drop(dfMeta[dfMeta.bad==True].index.to_list())
	dfRaw=_filterDFByTime(dfRaw,tStart,tStop)
		
	# filter data
	dfSmoothed=_gaussianFilter_df(dfRaw,timeFWHM=timeFWHMSmoothing,filterType='high',plot=False)
	
	# optional stripey plots
	if plot==True:
		if 'PA' in sensor:
			angle=dfMeta.theta.values
			dfSmoothed.columns=angle
			dfSmoothed.index*=1e3
			fig,ax,cax=_plotHbt.stripeyPlot(dfSmoothed*1e4,title='%d'%shotno,poloidal=True,subtitle=sensor,xlabel='Time (ms)',ylabel=r'Poloidal angle, $\theta$')
		elif sensor=='TA':
			angle=dfMeta.phi.values
			dfSmoothed.columns=angle
			dfSmoothed.index*=1e3
			fig,ax,cax=_plotHbt.stripeyPlot(dfSmoothed*1e4,title='%d'%shotno,toroidal=True,subtitle=sensor,xlabel='Time (ms)',ylabel=r'Toroidal angle, $\phi$')
		else: #FB arrays
			for s in ['S1P','S2P','S3P','S4P']:
				dfTemp=_filterDFByColOrIndex(dfSmoothed,s)
				dfTempMeta=_filterDFByColOrIndex(dfMeta,s,col=False)
				angle=dfTempMeta.phi.values
				dfTemp.columns=angle
				dfTemp.index*=1e3
				fig,ax,cax=_plotHbt.stripeyPlot(dfTemp*1e4,title='%d'%shotno,toroidal=True,subtitle='FB_'+s,xlabel='Time (ms)',ylabel=r'Toroidal angle, $\phi$')
			

	return dfRaw,dfSmoothed,dfMeta



def mModeData(	shotno,
				tStart=_TSTART,
				tStop=_TSTOP,
				sensor='PA1',
				modeNumbers=[0,2,3,4],
				plot=True):
	"""
	m mode analysis
		
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False	
	mode numbers : list of ints
		mode numbers to be analyzed
		[0,2,3,4] are the m=0,2,3,and 4 modes
	sensor : str
		Sensor to perform the analysis
		sensor should be in ['PA1','PA2']
	
	Returns
	-------
	dfResults : pandas.core.frame.DataFrame
		results of the m-mode analysis
	
	Examples
	--------
	::
		
		df=mModeData(100000,sensor='PA1')
		df=mModeData(100000,sensor='PA2',modeNumbers=[3])
	"""
	if sensor not in ['PA1','PA2']:
		raise Exception('Bad sensor name')
		
	_,df,dfMeta=magneticSensorData(shotno,
									tStart=tStart,
									tStop=tStop,
									sensor=sensor,
									forceDownload=False)

	angles=dfMeta.theta.values
	dfResults=_leastSquareModeAnalysis(	df*1e4,
									angles,
									modeNumbers=modeNumbers,
									plot=plot,
									title='m-mode analysis, %s, %d'%(sensor,shotno))
	
	return dfResults

	
def nModeData(	shotno,
				tStart=_TSTART,
				tStop=_TSTOP,
				sensor='TA',
				modeNumbers=[0,-1,-2],
				plot=False):
	"""
	n mode analysis
		
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False	
	mode numbers : list of ints
		mode numbers to be analyzed
		[0,-1,-2] are the n=0, 1,and 2 modes.
	sensor : str
		Sensor to perform the analysis
		sensor should be in ['TA','FBS1','FBS2','FBS3','FBS4']
	
	Returns
	-------
	dfResults : pandas.core.frame.DataFrame
		results of the n-mode analysis
	
	Examples
	--------
	::
		
		df=nModeData(100000,sensor='TA')
		df=nModeData(100000,sensor='FBS4')
		df=nModeData(100000,sensor='FB_S1',modeNumbers=[-1])
	"""
	
	if 'TA' in sensor:
		_,df,dfMeta=magneticSensorData(shotno,
										tStart=tStart,
										tStop=tStop,
										sensor='TA',
										forceDownload=False)
	elif 'FB' in sensor:
		_,df,dfMeta=magneticSensorData(shotno,
								tStart=tStart,
								tStop=tStop,
								sensor='FB',
								forceDownload=False)
		
		from johnspythonlibrary2.Process.Misc import extractIntsFromStr
		num=extractIntsFromStr(sensor)[0]
		if num not in [1,2,3,4]:
			raise Exception('Bad sensor name')
		from johnspythonlibrary2.Process.Pandas import filterDFByColOrIndex
		df=filterDFByColOrIndex(df,'S%dP'%num)
		dfMeta=filterDFByColOrIndex(dfMeta,'S%dP'%num,col=False)
	else:
		raise Exception('Bad sensor name')

	angles=dfMeta.phi.values
	dfResults=_leastSquareModeAnalysis(	df*1e4,
									angles,
									modeNumbers=modeNumbers,
									plot=plot,
									title='n-mode analysis, %s, %d'%(sensor,shotno))
	
	return dfResults




def plasmaRadiusData(	shotno=95782,
							tStart=_TSTART,
							tStop=_TSTOP, 
							plot=False,
							forceDownload=False):
	"""
	Calculate the major and minor radius.
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Notes
	-----
	The radius calculations below are pulled from Paul Hughes's 
	pauls_MDSplus_toolbox.py code.  In that code, he attributes Niko Rath for 
	its implementation.  I don't really understand it.
	
	Example
	-------
	::
		
		df=plasmaRadiusData(95782,plot=True)
	
	"""
	
	@_backupDFs
	def dfPlasmaRadius(shotno,
						tStart,
						tStop):
			
		# Determined by Daisuke during copper plasma calibration
		a=.00643005
		b=-1.10423
		c=48.2567
		
		# Calculated by Jeff, but still has errors
		vf_pickup = 0.0046315133 * -1e-3
		oh_pickup = 7.0723416e-08
		
		# get vf and oh data
		dfCapBank=capBankData(shotno,tStart=tStart,tStop=tStop)
		vf=dfCapBank.VF_CURRENT.to_numpy()
		oh=dfCapBank.OH_CURRENT.to_numpy()
		time=dfCapBank.index.to_numpy()
	
		# get plasma current
		dfIp=ipData(shotno,tStart=tStart,tStop=tStop)
		ip=dfIp.IpRog.to_numpy()*1212.3*1e-9  # ip gain
		
		# get cos-1 raw data
		dfCos1Rog=cos1RogowskiData(shotno,tStart=tStart,tStop=tStop) 
	
		# integrate cos-1 raw 
		from scipy.integrate import cumtrapz
		cos1=cumtrapz(dfCos1Rog.COS_1_RAW,dfCos1Rog.index)+dfCos1Rog.COS_1_RAW.iloc[:-1]*0.004571
		cos1=_np.append(cos1,0)
		
		# r-major calculations
		pickup = vf * vf_pickup + oh * oh_pickup
		ratio = ip / (cos1 - pickup)
		arg = b**2 - 4 * a * (c-ratio)
		arg[arg < 0] = 0
		r_major = (-b + _np.sqrt(arg)) / (2*a)
		majorRadius  = r_major / 100 # Convert to meters
		
		dfData=_pd.DataFrame()
		dfData['time']=time
		dfData['majorRadius']=majorRadius
		dfData=dfData.set_index('time')
		
		minorRadius=_np.ones(len(majorRadius))*0.15
		minorRadius[majorRadius>0.92]=0.15-(majorRadius[majorRadius>0.92]-0.92)
		minorRadius[majorRadius<0.9]=0.15-(0.9-majorRadius[majorRadius<0.9])
		dfData['minorRadius']=minorRadius
		
		return dfData
	
	dfData=dfPlasmaRadius(shotno,
					   tStart=tStart,
					   tStop=tStop,
					   forceDownload=forceDownload)
	dfData=_filterDFByTime(dfData,tStart,tStop)
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(dfData.majorRadius)
		ax[1].plot(dfData.minorRadius)
	
	return dfData


def qStarData(shotno=96496, tStart=_TSTART, tStop=_TSTOP, plot=False,forceDownload=False):
	"""
	Gets qstar data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
		
	Examples
	--------
	::
		
		qStarData(96496,plot=True)
	"""
	
	
	# get data
	dfIp=ipData(shotno,tStart=tStart,tStop=tStop,forceDownload=forceDownload)
	plasmaRadius=plasmaRadiusData(shotno,tStart=tStart,tStop=tStop,forceDownload=forceDownload)
	dfTF=tfData(shotno,tStart,tStop,forceDownload=forceDownload)
	
	
	tfProbeData=dfTF.TF.to_numpy()
	tfProbeData=tfProbeData*1.23/plasmaRadius.majorRadius
		
	# calc q star
	qStar= plasmaRadius.minorRadius**2 * tfProbeData / (2e-7 * dfIp.IpRog * plasmaRadius.majorRadius)
	qStarCorrected=qStar*(1.15) # 15% correction factor.  jeff believes our qstar measurement might be about 15% to 20% too low.  
	time=dfIp.index.to_numpy()
	
	def makePlot():
		""" 
		Plot all relevant plots 
		"""
		
		fig,p1=_plt.subplots()
		p1.plot(time*1e3,qStar,label=r'q$^*$')
		p1.plot(time*1e3,qStarCorrected,label=r'q$^* * 1.15$')
		_plot.finalizeSubplot(p1,xlabel='Time (ms)',ylabel=r'q$^*$',ylim=[1,5])
		_plot.finalizeFigure(fig,title='%s'%shotno)
		
	if plot == True:
		makePlot()
		
	dfData=_pd.DataFrame(qStar,index=time,columns=['qedge'])
	return dfData



def quartzJumperAndGroundingBusData(	shotno=96530,
										tStart=_TSTART,
										tStop=_TSTOP,
										plot=False,
										timeFWHMSmoothing=0.4e-3):
	"""
	External rogowski data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	timeFWHMSmoothing : float
		Time constant associated with the offset subtraction filter
		
		
		
	Notes
	-----
	Rog. D is permanently off for the time being
	Rog. C is permanently off for the time being
	
	"""
	
	# subfunctions
	@_backupDFs
	def dfJumperRaw(shotno, tStart, tStop,	dfJumperMeta):
		dfJumperRaw=mdsData(shotno,dfJumperMeta.addresses.to_list(), tStart, tStop,	columnNames=dfJumperMeta.index.to_list())
		return dfJumperRaw
	
	# load meta data
	sensor='Jumper'
	try:
		directory=_path.dirname(_path.realpath(__file__))
		dfMeta=_readOdsToDF('%s/listOfAllSensorsOnHBTEP.ods'%directory,sensor).set_index('names')
	except:
		dfMeta=_readOdsToDF('listOfAllSensorsOnHBTEP.ods',sensor).set_index('names')
	
	
	# load raw data
	dfRaw=dfJumperRaw(shotno,tStart,tStop,dfMeta)
	dfRaw['WestRackGround']*=100
	dfRaw['NorthRackGround']*=100
	dfRaw=_filterDFByTime(dfRaw,tStart,tStop)
	
	# filter data
	dfSmoothed=_gaussianFilter_df(dfRaw,timeFWHM=timeFWHMSmoothing,filterType='high',plot=False)

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		for i,(key,val) in enumerate(dfRaw.iteritems()):
			ax[0].plot(val.index*1e3,val,label=key)
		for i,(key,val) in enumerate(dfSmoothed.iteritems()):
			ax[1].plot(val.index*1e3,val,label=key)
		_plot.finalizeSubplot(	ax[0],
#								xlabel='Time',
								ylabel='A',
								subtitle='Raw',
								title='%d'%shotno,
#								ylim=[3,15],
#								legendOn=False,
								)
		_plot.finalizeSubplot(	ax[1],
								xlabel='Time',
								ylabel='A',
								subtitle='Smoothed',
								)
		
	return dfRaw,dfSmoothed,dfMeta
		


def spectrometerData(	shotno=98030,
						tStart=_TSTART,
						tStop=_TSTOP,
						plot=False):
	"""
	Spectrometer data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Example
	-------
	::
		
		df=spectrometerData(98030,plot=True)
	
	"""
	# get data
	df=mdsData(	shotno,
				dataAddress=['\HBTEP2::TOP.SENSORS.SPECTROMETER'],
				tStart=tStart, 
				tStop=tStop,
				columnNames=['spectrometer'])
	
	if plot == True:
		df.plot()
		
	return df

	
def solData(	shotno=98030,
				tStart=_TSTART,
				tStop=_TSTOP,
				plot=False,
				timeFWHMSmoothing=0.4e-3):
	"""
	SOL tile sensor data
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	timeFWHMSmoothing : float
		Time constant associated with the offset subtraction filter
		
	Returns
	-------
	dfRaw : pandas.core.frame.DataFrame
		Raw magnetic data.  Time is index.
	dfSmoothed : pandas.core.frame.DataFrame
		Offset subtracted magnetic data.  Time is index.
	dfMeta : pandas.core.frame.DataFrame
		Meta data for the sensors
	
	Examples
	--------
	::
		
		a,b,c=solData(98173)
	
	"""
	
	# subfunctions
	@_backupDFs
	def dfSOLRaw(shotno, tStart, tStop,	dfSOLMeta):
		dfSOLRaw=mdsData(shotno,dfSOLMeta.addresses.to_list(), tStart, tStop,	columnNames=dfSOLMeta.index.to_list())
		return dfSOLRaw
	
	# load meta data
	sensor='SOL'
	try:
		directory=_path.dirname(_path.realpath(__file__))
		dfMeta=_readOdsToDF('%s/listOfAllSensorsOnHBTEP.ods'%directory,sensor).set_index('names')
	except:
		dfMeta=_readOdsToDF('listOfAllSensorsOnHBTEP.ods',sensor).set_index('names')
	
	# load raw data
	dfRaw=dfSOLRaw(shotno,tStart,tStop,dfMeta)
	dfRaw=_filterDFByTime(dfRaw,tStart,tStop)
	
	# filter data
	dfSmoothed=_gaussianFilter_df(dfRaw,timeFWHM=timeFWHMSmoothing,filterType='high',plot=False)

	# plot
	if plot==True:
		dfRaw.plot()
		dfSmoothed.plot()
		
	return dfRaw,dfSmoothed,dfMeta


	

		
			


		






def sxrData(	shotno=98170,
				tStart=_TSTART,
				tStop=_TSTOP,
				plot=False,
				dropBadChannels=True,
				forceDownload=False):
	"""
	Downloads (and optionally plots) soft xray sensor data.   
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		default is False
	dropBadChannels : bool
		Drops bad channels
		
	Notes
	-----	
	Channels 5, 8, 10, 13 and 15 are considered "bad".  These particular channels
	are frequently "missing" from the tree, inlclude anamolous data, or their
	signals are attenuated.  
	
	Example
	-------
	::
		
		df=sxrData(106000,plot=True,tStart=1.5e-3,tStop=5.2e-3)

	"""
	
	# subfunctions
	@_backupDFs
	def dfSXR(shotno,  dfSXRMeta):
		dfSXR=mdsData(	shotno,
							dfSXRMeta.addresses.to_list(), 
							columnNames=dfSXRMeta.index.to_list())
		return dfSXR
	
	# load meta data
	sensor='SXR'
	try:
		directory=_path.dirname(_path.realpath(__file__))
		dfMeta=_readOdsToDF('%s/listOfAllSensorsOnHBTEP.ods'%directory,sensor).set_index('names')
	except:
		dfMeta=_readOdsToDF('listOfAllSensorsOnHBTEP.ods',sensor).set_index('names')
	
	# load raw data
	df=dfSXR(shotno,dfMeta,forceDownload=forceDownload)
	
	# drop bad channels
	if dropBadChannels==True:
		badSensors=dfMeta[dfMeta.bad==True].index.to_list()
		dfMeta=dfMeta.drop(badSensors,axis=0)
		df=df.drop(columns=badSensors)
		
	# trim time
	df=_filterDFByTime(df,tStart,tStop)
	
	# optional high-pass filter
	dfHP=_gaussianFilter_df(df,timeFWHM=0.4e-3)
	
	# optional plot
	if plot==True:
		
		if True: # raw data
			fig,ax,cax=_plot.subplotsWithColormaps(2,sharex=True)
			cax[0].remove()
			for key,val in df.iteritems():
				
				ax[0].plot(val,label=key)
			_plot.finalizeSubplot(ax[0],
											title='%d, raw'%shotno,)
			
			temp=_np.copy(df.columns).astype(str)
			columns=_np.array(['%d'%int(temp[i][-2:]) for i in range(len(temp))]).astype(float)
			df2=df.copy()
			df2.columns=columns
			fig,ax,_=_plotHbt.stripeyPlot(	df2,
											fig=fig,
											ax=ax[1],
											cax=cax[1],
	#										title='%d, raw'%shotno,
											colorMap='magma_r',
											zlim=[0,df.max().max()],
											levels=_np.linspace(0,df.max().max(),41),
											ylabel='Channel #')	
			_plot.finalizeFigure(fig)
			
		if True: # filtered data
			fig,ax,cax=_plot.subplotsWithColormaps(2,sharex=True)
			cax[0].remove()
			for key,val in dfHP.iteritems():
				
				ax[0].plot(val,label=key)
			_plot.finalizeSubplot(ax[0],
											title='%d, high-pass filtered'%shotno,)
			
			 
			temp=_np.copy(dfHP.columns).astype(str)
			columns=_np.array(['%d'%int(temp[i][-2:]) for i in range(len(temp))]).astype(float)
			dfHP2=dfHP.copy()
			dfHP2.columns=columns
			fig,ax,_=_plotHbt.stripeyPlot(dfHP2,
											fig=fig,
											ax=ax[1],
											cax=cax[1],
#								 title='%d, high-pass filtered'%shotno,
											ylabel='Channel #')		
			_plot.finalizeFigure(fig)
		
	return df,dfHP
			



def sxrMidplaneData(	shotno=96530,
						tStart=_TSTART,
						tStop=_TSTOP,
						plot=False):
	"""
	Gets Soft X-ray midplane sensor data at: devices.north_rack:cpci:input_74 
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
		
	Example
	-------
	::
		
		sxrMidplaneData(96530,plot=True)
	"""
	df=mdsData(shotno=shotno,
			   dataAddress=['\HBTEP2::TOP.DEVICES.NORTH_RACK:CPCI:INPUT_74 '],
			   tStart=tStart, tStop=tStop,
			   columnNames=['SXR_Midplane'])
	df.SXR_Midplane*=-1
	
	if plot==True:
		df.plot()
		
	return df


		
		
def tfData(	shotno=96530,
			tStart=_TSTART,
			tStop=_TSTOP,
			plot=False,
			upSample=True,
			forceDownload=False):
	"""
	Toroidal field data  
	
	Parameters
	----------
	shotno : int
		shot number of desired data
	tStart : float
		time (in seconds) to trim data before
		default is 0 ms
	tStop : float
		time (in seconds) to trim data after
		default is 10 ms
	plot : bool
		plots all relevant plots if true
		default is False
	upSample : bool
		up-samples the data to have a 2e-6 time step
		
	Notes
	-----
	note that the TF field data is recorded on an A14 where most of HBTEP data
	is stored with the CPCI.  Because the A14 has a slower sampling rate, this
	means that the TF data has fewer points than the rest of the HBTEP data, 
	and this makes comparing data difficult.  Therefore by default, I up-sample
	the data to match the CPCI sampling rate.  
	
	Example
	-------
	::
		
		df=tfData(96530,plot=True)
	"""
	
	@_backupDFs
	def dfTF(shotno,
			  tStart,
			  tStop):
		dfOld=mdsData(shotno=shotno,
				  dataAddress=['\HBTEP2::TOP.SENSORS.TF_PROBE'],
				  tStart=tStart, 
				  tStop=tStop,
				  columnNames=['TF']) 
		
		dt=2e-6
		time=_np.arange(tStart,tStop+dt,dt)
		x=_np.interp(time,dfOld.index.to_numpy(),dfOld.TF.to_numpy())
		df=_pd.DataFrame(x,index=time,columns=['TF'])
		
		df=df[(df.index>=tStart)&(df.index<=tStop)]
		
		return df
	
	df=dfTF(shotno,
		 tStart=tStart,
		 tStop=tStop,
		 forceDownload=forceDownload)
	df=_filterDFByTime(df,tStart,tStop)
	

	if plot==True:
		df.plot()
		
	return df
		
		
		










