
import numpy as _np
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Plot import subTitle as _subTitle, finalizeFigure as _finalizeFigure #, finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Spectral import fft_average, signal_spectral_properties
from johnspythonlibrary2.Process.Misc import subtract_mean_and_normalize_by_std, check_dims
import xarray as _xr

#%% Signal generators for bicoherence
# TODO add, y=cos(omega * t + 0.1*random(len(t)) )

# def generate_signals(dt, M, N, f, phase, seed=0,plot=False):
# 	"""
# 	dt=5e-1
# 	M=64     # number of windows
# 	N=128*4  # number of samples per window
# 	f=0.220*1*_np.pi*2
# 	seed=0
# 	phase=(_np.random.rand(M)-0.5)*_np.pi
# 	"""
# 	# M = number of windows
# 	# N = number of samples per window
# 	t = _np.arange(0, int(M*N)) * dt
# 	
# 	def sigGen(f=f, seed=seed, M=M, N=N, t=t):
# 		_np.random.seed(seed)
# 		theta = phase
# 		Theta = _np.repeat(theta[_np.newaxis,:], N, axis=0)
# 		
# # 		T=t.reshape((N,M))
# 		T=_np.array(list(t[0:N])*M).reshape(M,N).transpose()
# 		return _np.cos(2*_np.pi*f*T+Theta)
# 	
# 	y=sigGen()
# 	
# 	t = _np.arange(0, int(M*N)) * dt
# 	y=_xr.DataArray( y.transpose().flatten(), dims='t', coords=[t])
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		y.plot(ax=ax)
# 	
# 	return y


#%% main functions

def _bicoherence_helper(stft, f_units='Hz'):
	""" 
	subfunction for calculating the bicoherence 
	
	Parameters
	----------
	stft : xarray.DataArray
		This is the stft results just prior to calculating the bicoherence
	"""

	# calculate bicoherence numerator and denominators
	for i,ti in enumerate(stft.t.data): # TODO(John) Figure out how to vectorize this step
		#print(i,ti)
		b,FiFj,conjFij=bispectrum(stft.sel(t=ti),returnAll=True,plot=False,f_units=f_units)
		
		if i==0:
			numerator=FiFj*conjFij
			denom1=_np.abs(FiFj)**2
			denom2=_np.abs(conjFij)**2
		else:
			numerator+=FiFj*conjFij
			denom1+=_np.abs(FiFj)**2
			denom2+=_np.abs(conjFij)**2
			
	# finish bicoherence calc
	bicoh=numerator**2/(denom1.data*denom2.data)
	
	return bicoh


def bispectrum(	da,
				firstQuadrantOnly=False,
				plot=False,
				returnAll=False,
				f_units='Hz'):
	"""
	work in progress
	
	
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

	# convert indices to integers (makes it easier to index later on)
	index=da.f.copy().data
	df=index[1]-index[0]
	index=np.round(index/df).astype(int)
	
	# Solve for M1 = F(f1)*F(f2)
	def M1(signal,index,index1name='f2',index2name='f1'):
		return xr.DataArray(	np.outer(signal,signal),
				 			    dims=[index1name,index2name],
								coords={index1name:index,index2name:index})
	
	# Solve for M2 = F(f1+f2)
	def M2(signal,index,index1name='f2',index2name='f1'):
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
				 			    dims=[index1name,index2name],
								coords={index1name:f,index2name:f})
	
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
		
def bicoherence_2D(da, nperseg=500, verbose=True, precondition_signal=False, windowFunc='hann', m_range=_np.arange(-5,5.1,1, dtype=int)):
	"""
	
	References
	----------
	 * See Appendix A in https://aip.scitation.org/doi/10.1063/1.3429674
	 
	"""
	print("work in progress")
	
	
	
	
		
	import numpy as np
# 	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from scipy.signal.spectral import _spectral_helper
# 	import pandas as pd
	import xarray as xr
# 	import matplotlib.pyplot as plt
# 	from johnspythonlibrary2.Process.Misc import check_dims, subtract_mean_and_normalize_by_std
	from johnspythonlibrary2.Process.Spectral import signal_spectral_properties
	
	# make sure 't' and 'theta' are in dimensions
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	check_dims(da, dims=['t','theta'])
	
	# optional precondition signal
	if precondition_signal==True:
		da = subtract_mean_and_normalize_by_std(da,'t')

	dt,fsamp,fn,_,_=signal_spectral_properties(da,nperseg=nperseg,verbose=verbose).values()
	
	# Solve for the STFT results from each time window
	f,t,stft_results=_spectral_helper(	da.data,
										da.data,
										axis=0,
										fs=1/(da.t.data[1]-da.t.data[0]),
										window=windowFunc,
										nperseg=nperseg,
										noverlap=0,
										return_onesided=False,
										mode='stft')
	da2=xr.DataArray(stft_results, dims=['f',da.dims[1],'t'], coords=[f,da.coords['theta'].data,t]).sortby('f')
	
	# Solve for FFT (STFT with 1 window) of azimuthal
	dtheta=da2.coords['theta'].data[1]-da2.coords['theta'].data[0]
	m,t0,stft_results=_spectral_helper(	da2.data,
										da2.data,
										axis=1,
										fs=1/dtheta,
										window=windowFunc,
										nperseg=da2.shape[1],
										noverlap=0,
										return_onesided=False,
										mode='stft')
	m=np.round(m*np.pi*2).astype(int)
	X2D=xr.DataArray(stft_results[:,:,:,0], dims=['f','m','t'], coords=[f,m,t]).sortby('m')
# 	N=X2D.shape[0]
	
# 	signal_1 = X2D.
	
	def bispectrum_2D(signal_1, signal_2, signal_3, f_units='Hz'):
		
		f_old = signal_1.f.data
		df = f_old[1]-f_old[0]
		f_new = np.round(f_old/df).astype(int)
		
		# Solve for M1 = F(f1)*F(f2)
		def M1():
			return xr.DataArray(	np.outer(signal_1, signal_2),
					 			    dims=['f2','f1'],
									coords={'f2':f_old,'f1':f_old})
		
		# Solve for M2 = F(f1+f2)
		def M2():

			f=f_new
			x = signal_3.data
			N=X2D.shape[0]
			
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
									coords={'f2':f_old,'f1':f_old})
		
		# bispectrum
		m1=M1()
		m2=np.conj(M2())
		b=m1*m2
# 		b['f1']=b.f1.data*df
# 		b['f2']=b.f2.data*df
# 		m1['f1']=m1.f1.data*df
# 		m1['f2']=m1.f2.data*df
# 		m2['f1']=m2.f1.data*df
# 		m2['f2']=m2.f2.data*df
		b.f1.attrs["units"] = f_units
		b.f2.attrs["units"] = f_units
		m1.f1.attrs["units"] = f_units
		m1.f2.attrs["units"] = f_units
		m2.f1.attrs["units"] = f_units
		m2.f2.attrs["units"] = f_units
		
# 		if firstQuadrantOnly==True:
# 			b=b[b.f2>=0,b.f1>=0]
		
# 		if plot==True:
# 			_plt.figure()
# 			np.abs(b).plot()
			
# 		if returnAll==False:
		return b, m1, m2
# 		else:
# 			return b, m1, m2
		
	def _bicoherence_2D_helper2(X2D, m1, m2):
		# calculate bicoherence numerator and denominators
		m3=m1+m2
		
		if m3 in X2D.m.data:
			
			for i,ti in enumerate(X2D.t.data): # TODO(John) Figure out how to vectorize this step
				print(i,ti)
				signal_1 = X2D.sel(t=ti, m=m1)
				signal_2 = X2D.sel(t=ti, m=m2)
				signal_3 = X2D.sel(t=ti, m=m3)
				b,FiFj,conjFij=bispectrum_2D(signal_1, signal_2, signal_3)
				
				if i==0:
					numerator=FiFj*conjFij
					denom1=_np.abs(FiFj)**2
					denom2=_np.abs(conjFij)**2
				else:
					numerator+=FiFj*conjFij
					denom1+=_np.abs(FiFj)**2
					denom2+=_np.abs(conjFij)**2
					
			# finish bicoherence calc
			bicoh=numerator**2/(denom1.data*denom2.data)
			
			return bicoh.sortby('f1').sortby('f2')
		
		else:
			print('invalid m3 value')
			return []
		
	def _bicoherence_2D_helper1(X2D, m_range):
		
		m1_range,m2_range=np.meshgrid(m_range,m_range)
		m1_range=m1_range.reshape(-1)
		m2_range=m2_range.reshape(-1)
		
		b=_bicoherence_2D_helper2(X2D, 0, 0)
		
		data = xr.DataArray(np.zeros((b.shape[0], b.shape[1], len(m_range), len(m_range)), dtype=complex),
							  dims=['f2','f1','m1','m2'],
							  coords=[b.f2, b.f1, m_range.astype(int), m_range.astype(int)])
		for m1,m2 in zip(m1_range,m2_range):
			#m3=m1+m2
			#print(m1,m2,m3)
			
			b = _bicoherence_2D_helper2(X2D, m1, m2)
			if len(b)>0:
				data.loc[:,:,m1,m2]=b
			else:
				data.loc[:,:,m1,m2]=_np.nan
				
		return data
				
# 	m_range=np.arange(-4,4)
	result=_bicoherence_2D_helper1(X2D, m_range)
	
	return result


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

	dt,fsamp,fn,_,_=signal_spectral_properties(da,nperseg=nperseg,verbose=verbose).values()
	
	# Solve for the STFT results from each time window
	f,t,stft_results=_spectral_helper(	da.data,
										da.data,
										fs=1/(da.t.data[1]-da.t.data[0]),
										window=windowFunc,
										nperseg=nperseg,
										noverlap=0,
										return_onesided=False,
										mode='stft')
	
	df=pd.DataFrame(stft_results,index=f,columns=t) #TODO remove pandas step
	df.index.name='f'
	df.columns.name='t'
	da2=xr.DataArray(df).sortby('f')
	
	
	bicoh = _bicoherence_helper(da2, f_units=f_units)
	
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
		im=np.abs(bicoh).plot(ax=ax1,levels=np.linspace(0,np.abs(bicoh).max(),20+1),add_colorbar=False)

		# trick to get the subplots to line up correctly
		divider = make_axes_locatable(ax1)
		ax2 = divider.append_axes("bottom", size="50%", pad=.5,sharex = ax1)
		cax = divider.append_axes("right", size="5%", pad=0.08)
		cbar=plt.colorbar( im, ax=ax1, cax=cax, ticks= np.linspace(0,np.abs(bicoh).max(),6) )
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


def monteCarloNoiseFloorCalculation(da,
					nperseg,
					R=100,
					alpha=0.9,
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
	References
	----------
	 * code is based on this paper: http://arxiv.org/abs/1811.02973
	 """
	print('work in progress')
	
# 	nperseg=512
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

	params=signal_spectral_properties(da,nperseg=nperseg,verbose=verbose)
	
	# step 2 : generate bicoherence from actual data
	b=bicoherence(da, nperseg=nperseg, plot=plot,mask='None')
	
	
	
	# step 1a: Calculate fourier components from actual data for each time window 
	windowFunc='hann'
	f,t,stft_results=_spectral_helper(	da.data,
										da.data,
										fs=params['f_s'],
										window=windowFunc,
										nperseg=nperseg,
										noverlap=0,
										return_onesided=False,
										mode='stft')
	# step 1b: Calculate absolution value of each signal
	stft_results_abs=np.abs(stft_results)
	
	# step 3a: Generate random phase for each fourier component in each time window.  Do this R times.
	theta=np.random.rand(stft_results_abs.shape[0],stft_results_abs.shape[1],R)

	# step 3b: Simulate (monte carlo) fourier data using real amplitude and random phase
	stft_results_mc = np.repeat(stft_results_abs[:, :, np.newaxis], R, axis=2) * np.exp( 1j*theta)
	stft_results_mc=xr.DataArray(stft_results_mc,dims=['f','t','r'], coords=[f,t,np.arange(R)]).sortby('f')

	# step 3c: calculate bicoh for each r in range(R) to create the probability distribution function for each pair of frequency coordinates
	if True:  #TODO this is painfully slow.  figure out how to vectorize
		mc_results=xr.DataArray( np.zeros((stft_results_mc.f.shape[0], stft_results_mc.f.shape[0], R),dtype=complex))
		for i,ri in enumerate(range(R)):
			print('%d/%d'%(ri+1,R), end=' ')
			mc_results[:,:,i]=_bicoherence_helper(stft_results_mc[:,:,i])
		print('')
		mc_results=np.abs(mc_results)
	else:
		print('work in progress')
		# look into Numba functions 
		
	# step 4: alpha is already defined
	
	# step 5: Calculate critical bicoherence value (noise floor)
	from scipy.interpolate import interp1d
	def integrate_along_prob_function_integral_equals_value(x,y,alpha=0.9):
		# y(x) is the probability function
		y/=y.sum() # normalize such that integral is equal to 1
		p = np.cumsum(y) # p = cumalative sum of y(x)
		f=interp1d(p,x) # create interpolation of function, x(p)
		return float( f(alpha) ) # find where x(p) = alpha
			
	def hist_1d(a, alpha=0.9):
		if np.isnan(a).sum() > 0:
			return float(0)
		else:
			hist, bin_edges = np.histogram(a)
			hist = hist.astype(float)/hist.sum()
			bin_centers=(bin_edges[1:]+bin_edges[:-1])/2
			a= integrate_along_prob_function_integral_equals_value(x=bin_centers,y=hist,alpha=alpha)
			return a

	counts = np.apply_along_axis(hist_1d, axis=2, arr=mc_results.data, alpha=alpha)
	counts = xr.DataArray( counts,
						   dims=b.dims,
						   coords=b.coords )
	
	if plot==True:
		fig,ax=plt.subplots()
		counts.plot(ax=ax)
		ax.set_title('Critical bicoherence values')
	
	# Steps 6 and 7: Any place where bicoherence values are less than the critical bicoh values, set equal to NaN.  Otherwise, keep. 
	b_abs=np.abs(b).data
	b_abs[b_abs<counts.data]=np.nan
	b_final=xr.DataArray( b_abs, dims=b.dims, coords=b.coords)
	
	if plot==True:
		fig,ax=plt.subplots()
		b_final.plot(ax=ax)
	
	return b_final
# 	counts= counts.
	
#%% Examples


# def example_1_v3():
# 	
# 	M=64
# 	N=128*4
# 	dt=5e-1
# 	
# 	f_nyquist=1
# 	f_a=0.220*f_nyquist#*_np.pi*2
# 	f_b=0.375*f_nyquist#*_np.pi*2
# 	
# 	_np.random.seed(0)
# 	noise=_np.random.normal(0,0.1,(M,N)).flatten()*0
# 	
# 	_np.random.seed(1)
# 	phase_a=(_np.random.rand(M)-0.5)*_np.pi
# 	
# 	_np.random.seed(2)
# 	phase_b=(_np.random.rand(M)-0.5)*_np.pi
# 	
# 	baseSignal=generate_signals(dt=dt, M=M, N=N, f=f_a,phase=phase_a)+generate_signals(dt=dt, M=M, N=N, f=f_b,phase=phase_b)+noise
# 	
# 	nperseg=N
# 	bicoherence(baseSignal, nperseg=1000, plot=True)
	
	
# def example_1_v2(mask='A',fft_scale='linear'):
# 	
# 	
# 	
# 	M=64
# 	nperseg=128*4
# 	dt=5e-1
# 	t=np.arange(0,nperseg*M)*dt
# 	
# 	fN=1
# 	fa=0.220*fN*_np.pi*2
# 	fb=0.375*fN*_np.pi*2
# 	
# 	theta_a = 0
# 	theta_b = 0.3569
# 	
# 	np.random.seed(0)
# 	noise=np.random.normal(0,0.1,len(t))
# 	baseSignal=generate_signals(t,fa,theta_a)+generate_signals(t,fb,theta_b)+noise
# 	
# 	y=_xr.DataArray(baseSignal, dims='t',coords=[t])
	
def example_travelling_waves():
	
	import matplotlib.pyplot as plt
	import numpy as np
	import xarray as xr
	
	plt.close('all')
	
	### initialize examples
	numberRecords=64
	recordLength=128*4
	spatialPoints=100
	N=recordLength
	M=numberRecords
	O=spatialPoints
	dt=5e-1
	t=np.arange(0,N*M)*dt
	fN=1
	fa=0.220*fN
	fb=0.375*fN
	nperseg=recordLength
	x=np.linspace(0,2*np.pi,101)[:100]
	fN_x=0.5/(x[1]-x[0])
	ka=1
	kb=2
	
	def randomPhase(n=1,seed=0):
		np.random.seed(seed)
		return (np.random.rand(n)-0.5)*np.pi*2
	
	def sigGen(t,x,f,k,theta):
		M=len(theta)
		N=len(t)//M
		T,Theta,X=np.meshgrid(t[0:N],theta,x, indexing='ij')
		sig= xr.DataArray(1*np.cos(2*np.pi*T*f+Theta+k*X), dims=['N','M','X'], coords=[t[0:N], np.arange(M),x]).stack({'t':('M','N')}).transpose('t','X')
		return xr.DataArray(sig.data, dims=['t','theta'],coords=[t,x])
	
	thetaa=randomPhase(M,seed=1)
	thetab=randomPhase(M,seed=2)
	
	np.random.seed(0)
	noise=np.random.normal(0,0.1,(N*M,len(x)))
	
	baseSignal=sigGen(t,x,fa,ka,thetaa)+sigGen(t,x,fb,kb,thetab)+noise

	x3=baseSignal+0.5*sigGen(t,x,fa+fb,ka+kb,thetaa+thetab)
	bicoherence(	x3[:,0],
				nperseg=nperseg,
				windowFunc='hann',
# 				mask=mask,
				plot=True,
				fft_scale='linear',
# 				title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)$',
				)
	
# 	from johnspythonlibrary2.Process.Spectral import dispersion_plot
	from johnspythonlibrary2.HSVideo import dispersion_plot
	dispersion_plot(x3, nperseg_dim1=1000)
	
	result_2D=bicoherence_2D(x3,nperseg=nperseg)
	
# 	fig,ax=plt.subplots(4,4)
	for i,m1 in enumerate(np.arange(0,4)*1):
		for j,m2 in enumerate(np.arange(0,4)*-1):
			fig,ax=plt.subplots()
			np.abs(result_2D.sel(m1=m1,m2=m2, method='nearest')).plot(ax=ax, add_colorbar=True)
# 			np.abs(result_2D.sel(m1=m1,m2=m2, method='nearest')).plot(ax=ax[i,j], add_colorbar=False)
			ax.set_aspect('equal')	
# 			ax[i,j].set_aspect('equal')	
			

	
def example_stationary_waves(mask='A',fft_scale='linear'):
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
		return (np.random.rand(n)-0.5)*np.pi*2
	
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
# if __name__ == '__main__':
# 	example1(mask='none',fft_scale='linear')
