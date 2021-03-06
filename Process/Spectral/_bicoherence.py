

import matplotlib.pyplot as _plt
from johnspythonlibrary2.Plot import subTitle as _subTitle, finalizeFigure as _finalizeFigure #, finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Spectral import fft_average, signal_spectral_properties


#%% main functions
def bispectrum(	da,
				firstQuadrantOnly=False,
				plot=False,
				returnAll=False,
				f_units='Hz'):
	"""
	work in progress
	
	# TODO this code should be correct.  Double check, finalize function, and incorporate with bicoherence function()
	
	# TODO flip f_1 and f_2 on the bispectrum and bicoherence plots
	
	Examples
	--------
		
		
	Example 1::
		
		import xarray as xr
		import numpy as np; np.random.seed(2)
		import matplotlib.pyplot as plt; plt.close('all')
		from johnspythonlibrary2.Process.Spectral import fft
		
		## create signal and index
		t=np.arange(0,10e-3,10e-6)
		phase1,phase2=np.random.rand(2)*np.pi*2
		noise=np.random.normal(0,1,t.shape)
		f1=1e3
		f2=3.14159*1e3
		y=np.cos(2*np.pi*t*f1+phase1)*np.cos(2*np.pi*t*f2+phase2)+noise
		da=xr.DataArray(y,
					  dims=['t'],
					  coords={'t':t})
		da=fft(da,plot=True).sortby('f')
		b=bispectrum(da,plot=True,firstQuadrantOnly=True)
		
		y=0.5*np.cos(2*np.pi*t*(f2+f1)+phase1)+0.5*np.cos(2*np.pi*t*(f2-f1)+phase2)+noise
		da=xr.DataArray(y,
					  dims=['t'],
					  coords={'t':t})
		da=fft(da,plot=True).sortby('f')
		b=bispectrum(da,plot=True,firstQuadrantOnly=True)
		
	"""
	import numpy as np
	import xarray as xr
	
	N=da.shape[0]
	signal=da.data
	
	try:
		index=da.f.copy().data
	except:
		index=da.freq.copy().data
	df=index[1]-index[0]
	index=np.round(index/df).astype(int)
	
	# Solve for M1 = F(f1)*F(f2)
	def M1(signal,index,index1name='f2',index2name='f1'):
		return xr.DataArray(	np.outer(signal,signal),
				 			    dims=[index1name,index2name],
								coords={index1name:index,index2name:index})
	
	# Solve for M2 = F(f1+f2)
	def M2(signal,index):
		x=signal
		f=index
		
		# padding signal and index to account for the extremes of min(index)+min(index) and max(index)+max(index)
		f_long=np.arange(f.min()*2, 2*(f.max()+(f[1]-f[0])), f[1]-f[0])
		x_long=np.concatenate(	(np.zeros(N//2)*np.nan,
								  x,
								  np.zeros(int(np.ceil(N/2)))*np.nan))
		x_long=xr.DataArray(	x_long,
				 			    dims=['f'],
								coords={'f':f_long})
		
		# calculate each permutation of f1+f2
		f1temp,f2temp=np.meshgrid(f,f)
		fsumlist=(f1temp+f2temp).reshape(-1)
		
		# find M2 = F(f1+f2)
		return xr.DataArray(	x_long.loc[fsumlist].data.reshape(N,N),
				 			    dims=['f2','f1'],
								coords={'f2':f,'f1':f})
	
	# bispectrum
	m1=M1(signal=signal,index=index)
	m2=np.conj(M2(signal,index))
	b=m1*m2
	b['f1']=b.f1.data*df
	b['f2']=b.f2.data*df
	m1['f1']=m1.f1.data*df
	m1['f2']=m1.f2.data*df
	m2['f1']=m2.f1.data*df
	m2['f2']=m2.f2.data*df
	b.f1.attrs["units"] = f_units
	b.f2.attrs["units"] = f_units
	m1.f1.attrs["units"] = f_units
	m1.f2.attrs["units"] = f_units
	m2.f1.attrs["units"] = f_units
	m2.f2.attrs["units"] = f_units
	
	if firstQuadrantOnly==True:
		b=b[b.f2>=0,b.f1>=0]
	
	if plot==True:
		_plt.figure()
		np.abs(b).plot()
		
	if returnAll==False:
		return b
	else:
		return b, m1, m2




# def bicoherence_deprecated(	da,
# 					windowLength,
# 					numberWindows,
# 					plot=False,
# 					windowFunc='Hann',
# 					title='',
# 					mask='A',
# 					drawRedLines=[]):
# 	"""
# 	Bicoherence and bispectrum analysis.  This algorithm is based on [Kim1979].
# 	
# 	This code appears to be correct.  #TODO Verify its operation and then clean it up.
# 	
# 	http://electricrocket.org/2019/246.pdf
# 	
# 	Parameters
# 	----------
# 	sx : pandas.core.series.Series
# 		Signal.  index is time.
# 	windowLength : int
# 		Length of each data window
# 	numberWindows : int
# 		Number of data windows
# 	plot : bool
# 		Optional plot of data
# 	windowFunc : str
# 		'Hann' uses a Hann window (Default)
# 		Otherise, uses no window 
# 		
# 	Returns
# 	-------
# 	dfBicoh : pandas.core.frame.DataFrame
# 		Bicoherence results.  Index and columns are frequencies.
# 	dfBispec : pandas.core.frame.DataFrame
# 		Bispectrum results.  Index and columns are frequencies.
# 	
# 	References
# 	----------
# 	* Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979). 

# 	* D.Kong et al Nuclear Fusion 53, 113008 (2013).
# 	
# 	
# 	Examples
# 	--------
# 	Example set 1::
# 		
# 		import matplotlib.pyplot as plt
# 		import numpy as np
# 		import pandas as pd
# 		
# 		plt.close('all')
# 		### Example dataset.  Figure 4 in reference: Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979).
# 		
# 		### initialize examples
# 		numberRecords=64
# 		recordLength=128*2
# 		N=recordLength
# 		M=numberRecords
# 		dt=5e-1
# 		t=np.arange(0,N*M)*dt
# 		fN=1
# 		fb=0.220*fN
# 		fc=0.375*fN
# 		fd=fb+fc
# 		fa=fc-fb
# 		
# 		def randomPhase(n=1,seed=0):
# 			np.random.seed(seed)
# 			return (np.random.rand(n)-0.5)*np.pi
# 		
# 		def sigGen(t,f,theta):
# 			M=len(theta)
# 			N=len(t)//M
# 			T,Theta=np.meshgrid(t[0:N],theta)
# 			return 1*np.cos(2*np.pi*T*f+Theta)
# 		
# # 		def finalizeAndSaveFig(figName='',figSize=[6,4.5]):
# # 			fig=plt.gcf(); 
# # 			fig.axes[1].set_ylim([0,1.1]); 
# # 			fig.set_size_inches(figSize)
# # 			if figName!='':
# # 				fig.savefig(figName,dpi=150)
# # 				
# # 		def diagonalOverlay(f1,f0=[fa,fb,fc,fd],ax=None):
# # 			if type(ax)==type(None):
# # 				ax=plt.gcf().axes[0]
# # 			for f in f0:
# # 				x=f1[f1>=f/2.]
# # 				y=f-x
# # 				ax.plot(x,y,'r--',linewidth=0.5)
# 			
# 		thetab=randomPhase(M,seed=1)
# 		thetac=randomPhase(M,seed=2)
# 		noise=np.random.normal(0,0.1,(M,N))
# 		baseSignal=sigGen(t,fb,thetab)+sigGen(t,fc,thetac)+noise
# 	
# 		import xarray as xr
# 		
# 		### Figure 1
# 		x1=(baseSignal).flatten()
# 		da=xr.DataArray(x1,dims=['t'],coords={'t':t})
# 		dfBicoh=bicoherence(	da,
# 						windowLength=recordLength,
# 						numberWindows=numberRecords,
# 						windowFunc='Hann',
# 						mask='A',
# 						plot=True)
# 		_plt.gcf().savefig('images/figure1.png')
# 		
# 		### Figure 2
# 		thetad=randomPhase(M,seed=3)
# 		x2=(baseSignal+0.5*sigGen(t,fd,thetad)).flatten()
# 		da=xr.DataArray(x2,dims=['t'],coords={'t':t})
# 		dfBicoh=bicoherence(	da,
# 						windowLength=recordLength,
# 						numberWindows=numberRecords,
# 						windowFunc='Hann',
# 						mask='A',
# 						plot=True)
# 		_plt.gcf().savefig('images/figure2.png')
# 		
# 		### Figure 3
# 		x3=(baseSignal+0.5*sigGen(t,fd,thetab+thetac)).flatten()
# 		da=xr.DataArray(x3,dims=['t'],coords={'t':t})
# 		dfBicoh=bicoherence(	da,
# 						windowLength=recordLength,
# 						numberWindows=numberRecords,
# 						windowFunc='Hann',
# 						plot=True,
# 						mask='A',
# # 						drawRedLines=[fb,fc,fd],
# 						drawRedLines=[fd])
# 		_plt.gcf().savefig('images/figure3.png')
# 		
# 		### Figure 4
# 		x4=(baseSignal+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
# 		da=xr.DataArray(x4,dims=['t'],coords={'t':t})
# 		dfBicoh=bicoherence(	da,
# 						windowLength=recordLength,
# 						numberWindows=numberRecords,
# 						windowFunc='Hann',
# 						mask='A',
# 						plot=True,
# # 						drawRedLines=[fb,fc,fd,fa],
# 						drawRedLines=[fc,fd])
# 		_plt.gcf().savefig('images/figure4.png')
# 		
# 		### Figure 5
# 		x5=(baseSignal+0.5*sigGen(t,fd,thetad)+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
# 		da=xr.DataArray(x5,dims=['t'],coords={'t':t})
# 		dfBicoh=bicoherence(	da,
# 									windowLength=recordLength,
# 									numberWindows=numberRecords,
# 									windowFunc='Hann',
# 									mask='A',
# 									plot=True,
# 									drawRedLines=[fc,fd])
# 		_plt.gcf().savefig('images/figure5.png')


# 	Example 2 ::
# 		
# 		N=2e4
# 		x1,x2=jpl2.Process.EDM.coupledHarmonicOscillator(N=N,T=10,
# 								args=[1,10,1e-4],plot=True)
# 		numberWindows=40
# 		windowLength=N//numberWindows
# 		dfBicoh=bicoherence(	x1,
# 						windowLength=windowLength,
# 						numberWindows=numberWindows,
# 						windowFunc='Hann',
# 						mask='A',
# 						plot=True,
# 						drawRedLines=[73+16])
# 		
# # 	Example 3 ::
# # 		
# # 		t=np.arange(0,10e-3,2e-6)
# # 		f1=1.1234e4
# # 		f2=np.pi*1e4
# # 		f3=f1+f2
# # 		
# # 		np.random.seed()
# # 		
# # 		phi1=2*np.pi*f1*t+np.random.rand()*2*np.pi
# # 		phi2=2*np.pi*f2*t+np.random.rand()*2*np.pi
# # 		phi3=phi1+phi2+np.random.rand()*2*np.pi
# # 		
# # 		y1=np.sin(phi1)
# # 		y2=np.sin(phi2)
# # 		y3=np.sin(phi3)
# # 		y=y1+y2
# # 		
# # 		fig,ax=plt.subplots()
# # # 		ax.plot(t,y1,label='1')
# # # 		ax.plot(t,y2,label='2')
# # # 		ax.plot(t,y3,label='3')
# # 		ax.plot(t,y,label='3')
# # 		ax.legend()
# # 		
# # 		da=_xr.DataArray(y,
# # 					   dims=['t'],
# # 					   coords={'t':t})
# # 		bicoherence(da, 510,9,plot=True)

# 	"""
# 	import numpy as np
# # 	import matplotlib.pyplot as plt
# # 	import pandas as pd
# # 	import xarray as xr
# 	from mpl_toolkits.axes_grid1 import make_axes_locatable
# 	
# 	N=int(windowLength)
# 	M=int(numberWindows)
# 	
# 	try:
# 		time=np.array(da.t)
# 	except:
# 		time=np.array(da.time)
# 	dt=time[1]-time[0]
# 	fsamp=1.0/dt
# 	
# 	print("frequency resolution : %.3f"%(fsamp*M/da.shape[0]))
# 	print("nyquist freqency : %.3f"%(fsamp/2))
# 		
# 	### main code
# 	
# 	# calculate window function
# 	if windowFunc=='Hann':
# 		n=np.arange(0,N)
# 		window=np.sin(np.pi*n/(N-1.))**2
# 		window/=np.sum(window)*1.0/N  	# normalize
# 	else:
# 		raise Exception('Hann is the only valid window function at the moment')
# 		
# 	# step in time
# 	for i in range(M):
# # 		print(i)
# 		# window data
# 		index=np.arange(N*(i),N*(i+1),)
# 		da_xi=(da[index]-da[index].mean())*window
# 		
# 		if i==0:
# 			p=True
# 		else:
# 			p=False
# 		# fft 
# 		da_Xi=fft(da_xi,plot=p,sortFreqIndex=True)
# # 		import sys;sys.exit()
# 		# bispectrum
# 		b,FiFj,conjFij=bispectrum(da_Xi,returnAll=True,plot=p)
# 		
# 		# calculate bicoherence numerator and denominators
# 		if i==0:
# 			numerator=FiFj*conjFij
# 			denom1=np.abs(FiFj)**2
# 			denom2=np.abs(conjFij)**2
# 		else:
# 			numerator+=FiFj*conjFij
# 			denom1+=np.abs(FiFj)**2
# 			denom2+=np.abs(conjFij)**2
# 			
# 	# calculate bicoherence
# 	bicoh=numerator**2/(denom1*denom2)
# 	bicoh['f1']=b.f1
# 	bicoh['f2']=b.f2
# 	
# 	# options
# 	if mask=='AB':
# 		f1=bicoh.coords['f1']
# 		f2=bicoh.coords['f2']
# 		
# 		a=(f1<=f2)&(f1>=-f2)
# 		b=(a*1.0).values
# 		b[b==0]=np.nan
# 		bicoh*=b
# 		bicoh=bicoh[:,bicoh.f2>=0]
# 		bicoh=bicoh[bicoh.f1<=f1.max()/2,:]
# 	elif mask=='A':
# 		f1=bicoh.coords['f1']
# 		f2=bicoh.coords['f2']
# 		
# 		a=(f1<=f2)&(f1>=-f2)
# 		b=(a*1.0).values
# 		b[b==0]=np.nan
# 		bicoh*=b
# 		bicoh=bicoh[:,bicoh.f2>=0]
# 		bicoh=bicoh[bicoh.f1>=0,:]
# 		bicoh=bicoh[bicoh.f1<=f1.max()/2,:]
# 	else:
# 		raise Exception('Improper mask value encountered : %s'%(str(mask)))
# 		
# 	bicoh.f1.attrs['units']='Hz'
# 	bicoh.f2.attrs['units']='Hz'
# 		
# 	if plot==True:
# 		fig,ax=_plt.subplots(2,1,sharex=False)
# 		
# 		im=np.abs(bicoh).plot(ax=ax[0],levels=np.linspace(0,1,20+1))
# 		ax[0].set_aspect('equal')
# 		fig.get_axes()[-1].remove()
# 		divider = make_axes_locatable(ax[0])
# 		cax = divider.append_axes("right", size="3%", pad=0.1)
# 		_plt.colorbar(im, cax=cax)#,label='Bicoherence')
# 		_subTitle(ax[0],'Bicoherence',
# 				xy=(0.98, .98),
# 				horizontalalignment='right',)
# 		
# 		fft_results=fft_average(da,nperseg=windowLength,noverlap=windowLength//2,trimNegFreqs=True,sortFreqIndex=True)
# 		fft_results/=fft_results.sum()
# 		np.abs(fft_results).plot(ax=ax[1],yscale='log')
# 		_subTitle(ax[1],'FFT')
# 		divider2 = make_axes_locatable(ax[1])
# 		cax2 = divider2.append_axes("right", size="3%", pad=0.1)
# 		cax2.remove()
# 		
# 		
# 		for y0 in drawRedLines:
# 			f1=bicoh.coords['f1'].data
# 			f2=bicoh.coords['f2'].data
# 			ax[0].plot(f2,y0-f2,color='r',linestyle='--',linewidth=0.5)
# 	
# 		fig.set_size_inches(4,4)
# 		
# 		
# 	return bicoh
			


def bicoherence(	da,
					nperseg,
					plot=False,
					windowFunc='hann',
					title='',
					mask='A',
					drawRedLines=[],
					f_units='Hz',
					verbose=True,
					fft_scale='log',
					precondition_signal=True):
	"""
	Bicoherence and bispectrum analysis.  This algorithm is based on [Kim1979].
	
	This code appears to be correct.  #TODO Verify its operation and then clean it up.
	
	
	Parameters
	----------
	sx : xarray.core.dataarray.DataArray
		Signal.  dim is 't' = time
	nperseg : int
		Length of each data window (number of points, N, per time segment)
	plot : bool
		Optional plot of data
	windowFunc : str
		'hann' uses a Hann window (Default)
	title : str
		Optional title for the resulting plot
	drawRedLines : list of floats
		List of floats correpsonding to f3=f1+f2 lines to be drawn
	f_units : str
		Unit label for the frequency results
	verbose : bool
		Prints misc Fourier details to screen 
		
	Returns
	-------
	bicoh : pandas.core.frame.DataFrame
		Bicoherence results.  Dimensions are frequencies, f1 and f2

	References
	----------
	* Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979). 

	* D.Kong et al Nuclear Fusion 53, 113008 (2013).
	
	* http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?bibcode=1995ESASP.371...61K&db_key=AST&page_ind=1&plate_select=NO&data_type=GIF&type=SCREEN_GIF&classic=YES - alternative bicoherence normaliziation
	
	Examples
	--------
	Example set 1::
		
		import matplotlib.pyplot as plt
		import numpy as np
		import pandas as pd
		
		plt.close('all')
		### Example dataset.  Figure 4 in reference: Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979).
		
		### initialize examples
		numberRecords=64
		recordLength=128*2
		N=recordLength
		M=numberRecords
		dt=5e-1
		t=np.arange(0,N*M)*dt
		fN=1
		fb=0.220*fN
		fc=0.375*fN
		fd=fb+fc
		fa=fc-fb
		
		def randomPhase(n=1,seed=0):
			np.random.seed(seed)
			return (np.random.rand(n)-0.5)*np.pi
		
		def sigGen(t,f,theta):
			M=len(theta)
			N=len(t)//M
			T,Theta=np.meshgrid(t[0:N],theta)
			return 1*np.cos(2*np.pi*T*f+Theta)
		
# 		def finalizeAndSaveFig(figName='',figSize=[6,4.5]):
# 			fig=plt.gcf(); 
# 			fig.axes[1].set_ylim([0,1.1]); 
# 			fig.set_size_inches(figSize)
# 			if figName!='':
# 				fig.savefig(figName,dpi=150)
# 				
# 		def diagonalOverlay(f1,f0=[fa,fb,fc,fd],ax=None):
# 			if type(ax)==type(None):
# 				ax=plt.gcf().axes[0]
# 			for f in f0:
# 				x=f1[f1>=f/2.]
# 				y=f-x
# 				ax.plot(x,y,'r--',linewidth=0.5)
			
		thetab=randomPhase(M,seed=1)
		thetac=randomPhase(M,seed=2)
		noise=np.random.normal(0,0.1,(M,N))
		baseSignal=sigGen(t,fb,thetab)+sigGen(t,fc,thetac)+noise
	
		import xarray as xr
		
		### Figure 1
		x1=(baseSignal).flatten()
		da=xr.DataArray(x1,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='Hann',
						mask='A',
						plot=True)
		_plt.gcf().savefig('images/figure1.png')
		
		### Figure 2
		thetad=randomPhase(M,seed=3)
		x2=(baseSignal+0.5*sigGen(t,fd,thetad)).flatten()
		da=xr.DataArray(x2,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='Hann',
						mask='A',
						plot=True)
		_plt.gcf().savefig('images/figure2.png')
		
		### Figure 3
		x3=(baseSignal+0.5*sigGen(t,fd,thetab+thetac)).flatten()
		da=xr.DataArray(x3,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='Hann',
						plot=True,
						mask='A',
# 						drawRedLines=[fb,fc,fd],
						drawRedLines=[fd])
		_plt.gcf().savefig('images/figure3.png')
		
		### Figure 4
		x4=(baseSignal+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
		da=xr.DataArray(x4,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='Hann',
						mask='A',
						plot=True,
# 						drawRedLines=[fb,fc,fd,fa],
						drawRedLines=[fc,fd])
		_plt.gcf().savefig('images/figure4.png')
		
		### Figure 5
		x5=(baseSignal+0.5*sigGen(t,fd,thetad)+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
		da=xr.DataArray(x5,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
									windowFunc='Hann',
									mask='A',
									plot=True,
									drawRedLines=[fc,fd])
		_plt.gcf().savefig('images/figure5.png')


	Example 2 ::
		
		from scipy.fftpack import next_fast_len
		_plt.close('all')
		from johnspythonlibrary2.Process.SigGen import coupledHarmonicOscillator_nonlinear
# 		import polycoherence as plc
		x1,x2=coupledHarmonicOscillator_nonlinear(	N=int(115*512/2),
													dt=1/2e3,
													plot=True,
													verbose=False,
													args={	'f1':45,
															'f2':150,
															'm':1,
															'E':3e3},).values()
# 		N=x1.shape[0]
# 		kw = dict(nperseg=N // 200, noverlap=0, nfft=next_fast_len(N // 2))
# 		fs=1/(x1.t.data[1])
# 		freq1, freq2, bicoh = plc.polycoherence.polycoherence(x1.data, fs=fs, **kw)
# 		import xarray as xr
# 		b=xr.DataArray( 	bicoh.real,
# 							 dims=['f1','f2'],
# 							 coords=[freq1,freq2])
# 		_plt.figure();b.plot()
# 		plc.polycoherence.plot_polycoherence(freq1, freq2, bicoh)

# 		dfBicoh=bicoherence(	x1,
# 						nperseg=502,
# 						windowFunc='Hann',
# 						mask='A',
# 						plot=True,
# # 						drawRedLines=[73+16],
# 						)
		
# 	Example 3 ::
# 		
# 		t=np.arange(0,10e-3,2e-6)
# 		f1=1.1234e4
# 		f2=np.pi*1e4
# 		f3=f1+f2
# 		
# 		np.random.seed()
# 		
# 		phi1=2*np.pi*f1*t+np.random.rand()*2*np.pi
# 		phi2=2*np.pi*f2*t+np.random.rand()*2*np.pi
# 		phi3=phi1+phi2+np.random.rand()*2*np.pi
# 		
# 		y1=np.sin(phi1)
# 		y2=np.sin(phi2)
# 		y3=np.sin(phi3)
# 		y=y1+y2
# 		
# 		fig,ax=plt.subplots()
# # 		ax.plot(t,y1,label='1')
# # 		ax.plot(t,y2,label='2')
# # 		ax.plot(t,y3,label='3')
# 		ax.plot(t,y,label='3')
# 		ax.legend()
# 		
# 		da=_xr.DataArray(y,
# 					   dims=['t'],
# 					   coords={'t':t})
# 		bicoherence(da, 510,9,plot=True)

	"""
	import numpy as np
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from scipy.signal.spectral import _spectral_helper
	import pandas as pd
	import xarray as xr
	import matplotlib.pyplot as plt
	
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'%(str(da.dims)))
		
	if precondition_signal==True:
		da=(da.copy()-da.mean(dim='t').data)/da.std(dim='t').data

	dt,fsamp,fn,_,_,_=signal_spectral_properties(da,nperseg=nperseg,verbose=verbose).values()
	
	# Solve for the STFT results from each time window
	f,t,stft_results=_spectral_helper(	da.data,
										da.data,
										fs=1/(da.t.data[1]-da.t.data[0]),
										window=windowFunc,
										nperseg=nperseg,
										noverlap=0,
										return_onesided=False,
										mode='stft')
	
	df=pd.DataFrame(stft_results,index=f,columns=t)
	df.index.name='f'
	df.columns.name='t'
	da2=xr.DataArray(df).sortby('f')
		
	# calculate bicoherence numerator and denominators
	for i,ti in enumerate(da2.t.data): # TODO(John) Figure out how to vectorize this step
		#print(i,ti)
		b,FiFj,conjFij=bispectrum(da2.sel(t=ti),returnAll=True,plot=False,f_units=f_units)
		
		if i==0:
			numerator=FiFj*conjFij
			denom1=np.abs(FiFj)**2
			denom2=np.abs(conjFij)**2
		else:
			numerator+=FiFj*conjFij
			denom1+=np.abs(FiFj)**2
			denom2+=np.abs(conjFij)**2
			
	# finish bicoherence calc
	bicoh=numerator**2/(denom1.data*denom2.data)
	
	# options
	if mask=='AB':
		f1=bicoh.coords['f1']
		f2=bicoh.coords['f2']
		
		a=(f1<=f2)&(f1>=-f2)
		b=(a*1.0).values
		b[b==0]=np.nan
		bicoh*=b
		#bicoh=bicoh[bicoh.f2>=0,:]
		bicoh=bicoh[:,bicoh.f1>=0]
		bicoh=bicoh[bicoh.f2<=fn/2]
	elif mask=='A':
		f1=bicoh.coords['f1']
		f2=bicoh.coords['f2']
		
		a=(f2<=f1)&(f2>=-f1)
		b=(a*1.0).values
		b[b==0]=np.nan
		bicoh*=b
		bicoh=bicoh[bicoh.f2>=0,:]
		bicoh=bicoh[:,bicoh.f1>=0]
		bicoh=bicoh[bicoh.f2<=fn/2]
	elif mask=='none' or mask=='None':
		pass
	else:
		raise Exception('Improper mask value encountered : %s'%(str(mask)))
		
	bicoh.f1.attrs['units']=f_units
	bicoh.f2.attrs['units']=f_units
	
	if plot==True:
		
		fig = plt.figure(  )
		
		# subplot 1
		ax1 = plt.subplot(111)
		ax1.set_aspect('equal')
		im=np.abs(bicoh).plot(ax=ax1,levels=np.linspace(0,1,20+1),add_colorbar=False)

		# trick to get the subplots to line up correctly
		divider = make_axes_locatable(ax1)
		ax2 = divider.append_axes("bottom", size="50%", pad=.5,sharex = ax1)
		cax = divider.append_axes("right", size="5%", pad=0.08)
		cbar=plt.colorbar( im, ax=ax1, cax=cax, ticks= np.linspace(0,1,6) )
		cbar.set_label('$b^2$')
		
		# subplot 2
		fft_results=fft_average(	da,
									nperseg=nperseg,
									noverlap=0,
									trimNegFreqs=False,
									sortFreqIndex=True,
									f_units=f_units,
									realAmplitudeUnits=True)
		# fft_results/=fft_results.sum() # normalize
		np.abs(fft_results).plot(ax=ax2,yscale=fft_scale)
		ax2.set_ylabel('Spectral density (au)')
		
		# optional, draw red lines
		for y0 in drawRedLines:
			f1=bicoh.coords['f1'].data
			f2=bicoh.coords['f2'].data
			ax1.plot(f1,y0-f1,color='r',linestyle='--',linewidth=0.5)
	
		# finalize
		_subTitle(ax1,'Bicoherence',
				xy=(0.98, .98),
				horizontalalignment='right',)
		_subTitle(ax2,'FFT')
		fig.set_size_inches(6,4)
		ax1.set_title(title)
		_finalizeFigure(fig)
			
		
	return bicoh
			

#%% Examples
def example1(mask='A',fft_scale='linear'):
	###  Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979).
	
	
	import matplotlib.pyplot as plt
	import numpy as np
	import xarray as xr
	
	plt.close('all')
	
	### initialize examples
	numberRecords=64
	recordLength=128*4
	N=recordLength
	M=numberRecords
	dt=5e-1
	t=np.arange(0,N*M)*dt
	fN=1
	fa=0.220*fN
	fb=0.375*fN
	nperseg=recordLength
	
	def randomPhase(n=1,seed=0):
		np.random.seed(seed)
		return (np.random.rand(n)-0.5)*np.pi
	
	def sigGen(t,f,theta):
		M=len(theta)
		N=len(t)//M
		T,Theta=np.meshgrid(t[0:N],theta)
		return 1*np.cos(2*np.pi*T*f+Theta)
	
	thetaa=randomPhase(M,seed=1)
	thetab=randomPhase(M,seed=2)
	
	np.random.seed(0)
	noise=np.random.normal(0,0.1,(M,N))
	
	baseSignal=sigGen(t,fa,thetaa)+sigGen(t,fb,thetab)+noise

	
	### Figure 1
	x1=(baseSignal).flatten()
	da=xr.DataArray(x1,dims=['t'],coords={'t':t})
	dfBicoh=bicoherence(	da,
					nperseg=nperseg,
					windowFunc='hann',
					mask=mask,
					plot=True,
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)$')
	_plt.gcf().savefig('images/figure1.png')
	
	### Figure 2
	thetac=randomPhase(M,seed=3)
	x2=(baseSignal+0.5*sigGen(t,fa+fb,thetac)).flatten()
	da=xr.DataArray(x2,dims=['t'],coords={'t':t})
	dfBicoh=bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					mask=mask,
					plot=True,
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+\frac{1}{2}cos(2 \pi (f_a+f_b) t + \theta_c)$')
	_plt.gcf().savefig('images/figure2.png')
	
	### Figure 3
	x3=(baseSignal+0.5*sigGen(t,fa+fb,thetaa+thetab)).flatten()
	da=xr.DataArray(x3,dims=['t'],coords={'t':t})
	dfBicoh=bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					plot=True,
					mask=mask,
# 						drawRedLines=[fb,fc,fd],
					drawRedLines=[fa+fb],
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+\frac{1}{2}cos(2 \pi (f_a+f_b) t + \theta_a+\theta_b)$')
	_plt.gcf().savefig('images/figure3.png')
	
	### Figure 4
	x4=(baseSignal+1*sigGen(t,fa,thetaa)*sigGen(t,fb,thetab)).flatten()
	da=xr.DataArray(x4,dims=['t'],coords={'t':t})
	dfBicoh=bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					mask=mask,
					plot=True,
# 						drawRedLines=[fb,fc,fd,fa],
					drawRedLines=[fa+fb,fb],
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+cos(2 \pi f_a t + \theta_a)cos(2 \pi f_b t + \theta_b)$')
	_plt.gcf().savefig('images/figure4.png')
	
	### Figure 5
	x5=(baseSignal+0.5*sigGen(t,fa+fb,thetac)+1*sigGen(t,fa,thetaa)*sigGen(t,fb,thetab)).flatten()
	da=xr.DataArray(x5,dims=['t'],coords={'t':t})
	dfBicoh=bicoherence(	da,
							nperseg=recordLength,
							windowFunc='hann',
							mask=mask,
							plot=True,
							drawRedLines=[fa+fb,fb],
							fft_scale='linear',
							title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+cos(2 \pi (f_a+f_b) t + \theta_c)+cos(2 \pi f_a t + \theta_a)cos(2 \pi f_b t + \theta_b)$')
	_plt.gcf().savefig('images/figure5.png')

#%% main
if __name__ == '__main__':
	example1(mask='none',fft_scale='linear')