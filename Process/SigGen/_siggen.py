
import numpy as _np
#import pandas as _pd
import matplotlib.pyplot as _plt
#from johnspythonlibrary2 import Plot as _plot
from johnspythonlibrary2.Process.Misc import findNearest
import xarray as _xr


#%% Basic signal generation

def squareWave(	freq,
				time,
				duty_cycle=0.5,
				plot=False):
	"""
	
	Examples
	--------
	Example 1::
		
		freq=1e2
		dt=1e-5
		time=np.arange(10000)*dt
		
		fig,ax=plt.subplots()
		ax.plot(	squareWave(freq,time))
	"""
	from scipy import signal as sg
	
	y = _xr.DataArray(	sg.square(2*_np.pi*freq*time, duty=duty_cycle),
						dims=['t'],
						coords={'t':time})
	
	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(y)
		
	return y


def chirp(	t,
			tStartStop=[0,1],
			fStartStop=[1e3,1e4],
			phi=270,
			plot=False,
			method='logarithmic'):
	"""
	Creates an exponential (also called a "logarithmic") chirp waveform
	
	Wrapper for scipy function.
	
	Example
	-------
	::
	
		import numpy as np
		t=np.arange(0,.005,6e-6);
		f0=1e3
		f1=2e4
		t0=1e-3
		t1=4e-3
		y=chirp(t,[t0,t1],[f0,f1],plot=True)
		
	"""
	from scipy.signal import chirp
	
	# find range of times to modify
	iStart=findNearest(t,tStartStop[0])
	iStop=findNearest(t,tStartStop[1])
	
	# create chirp signal using scipy function.  times are (temporarily) shifted such that the phase offset, phi, still makes sense
	y=_np.zeros(len(t))
	y[iStart:iStop]=chirp(t[iStart:iStop]-tStartStop[0],fStartStop[0],tStartStop[1]-tStartStop[0],fStartStop[1],method,phi=phi)
	y= _xr.DataArray(	y,
						dims=['t'],
						coords={'t':t})

	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(y)
		
	return y


def gaussianNoise(t,mean=0,stdDev=1,plot=False):
	"""
	Produces Gaussian noise based on the standard deviation
	This is a wrapper for the numpy.random.normal function

	Parameters
	----------
	shape : tuple of ints
		Shape of desired noise; such as generated from np.shape()
	mean : float
		Offset value. The default is 0.
	stdDev : float
		"Amplitude" (standard deviation) of the noise. The default is 1.
	plot : bool, optional
		Optional plot of results

	Returns
	-------
	noise : numpy array
		Array containing the generated noise.  
		
	Examples
	--------
	
	Example 1::
		
		t=np.arange(0,10e-3,2e-6)
		y1=np.sin(2*np.pi*t*1e3)
		y2=y1+gaussianNoise(t,mean=0,stdDev=0.5)
		fig,ax=plt.subplots()
		ax.plot(t,y2,label='signal with noise')
		ax.plot(t,y1,label='signal without noise')
		ax.legend()

	"""
	
	noise = _xr.DataArray(	_np.random.normal(mean,stdDev,t.shape),
							dims=['t'],
							coords={'t':t})
	
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(noise)
		
	return noise


#%% More complicated functions

def tentMap(N=1000,plot=False,ICs={'x0':_np.sqrt(2)/2.0}):
	""" 
	generate fake data using a tentmap 
	
	x=tentMap(1000,plot=True)
	
	References
	----------
	* https://en.wikipedia.org/wiki/Tent_map
	"""
	t=_np.arange(0,N+1)
	x=_np.zeros(t.shape,dtype=_np.float64)
	x[0]=ICs['x0']
	def tentMap(x,mu=_np.float64(2-1e-15)):
		for i in range(1,x.shape[0]):
			if x[i-1]<0.5:
				x[i]=mu*x[i-1]
			elif x[i-1]>=0.5:
				x[i]=mu*(1-x[i-1])
			else:
				raise Exception('error')
		return x
	
	x=tentMap(x)
	
	# the actual data of interest is the first difference of x
	xDelta=_xr.DataArray(	x[1:]-x[0:-1],
						dims=['t'],
						coords={'t':t[1:]},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	xDelta.t.attrs={'units':'au'}
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(t,x,linestyle='-',marker='.',linewidth=0.5,markersize=3,label='x')
		xDelta.plot(ax=ax[1],linestyle='-',marker='.',linewidth=0.5,markersize=3,label='x derivative')
		ax[0].legend()
		ax[1].legend()
		
	return xDelta


def lorentzAttractor(	N=2000,
						dt=0.05,
						ICs={	'x0':-9.38131377,
							    'y0':-8.42655716 , 
								'z0':29.30738524},
						plot=False,
						removeMean=False,
						normalize=False,
						removeFirstNPoints=500,
						args={	'sigma':10.0,
								'b':8.0/3.0,
								'r':28.0}):
	"""
	Solves the lorentz attractor nonlinear ODEs.
	
	References
	----------
	 * https://en.wikipedia.org/wiki/Lorenz_system
	 
	Examples
	--------
	Example 1::
		
		_plt.close('all')
		x,y,z=solveLorentz( 	N=5000,
						dt=0.02,
						IC=[-9.38131377, -8.42655716 , 29.30738524],
						plot='all',
						removeMean=True,
						normalize=True,
						removeFirstNPoints=500)
						
	"""
	
	from scipy.integrate import solve_ivp
	
	## initialize
	N+=removeFirstNPoints
	T=N*dt
	ICs=_np.array(list(ICs.items()))[:,1].astype(float)
	
	## Solve Lorentz system of equations
	def ODEs(t,y,*args):
		X,Y,Z = y
		sigma,b,r=args
		derivs=	[	sigma*(Y-X),
					-X*Z+r*X-Y,
					X*Y-b*Z]
		return derivs
	
	t_eval=_np.arange(0,T,dt)
	psoln = solve_ivp(	ODEs,
						[0,T],
						ICs,  # initial conditions
						args=args.values(),
						t_eval=t_eval
						)
	x,y,z=psoln.y
	
	## Cleanup data
	x=_xr.DataArray(	x,
						dims=['t'],
						coords={'t':t_eval},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x.t.attrs={'units':'au'}
	y=_xr.DataArray(	y,
						dims=['t'],
						coords={'t':t_eval},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	y.t.attrs={'units':'au'}
	z=_xr.DataArray(	z,
						dims=['t'],
						coords={'t':t_eval},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	z.t.attrs={'units':'au'}
	
	if removeFirstNPoints>0:
		x=x[removeFirstNPoints:]
		y=y[removeFirstNPoints:]
		z=z[removeFirstNPoints:]
	
	if removeMean==True:
		x-=x.mean()
		y-=y.mean()
		z-=z.mean()
		
	if normalize==True:
		x/=x.std()
		y/=y.std()
		z/=z.std()
		
	ds=_xr.Dataset({	'x':x,
						'y':y,
						'z':z})
			
	## optional plots
	if plot!=False:
		fig,ax=_plt.subplots(3,sharex=True)
		markersize=2
		x.plot(ax=ax[0],marker='.',markersize=markersize)
		y.plot(ax=ax[1],marker='.',markersize=markersize)
		z.plot(ax=ax[2],marker='.',markersize=markersize)
		ax[0].set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args['sigma'],args['b'],args['r'])+'\nICs = (%.3f, %.3f, %.3f)'%(ICs[0],ICs[1],ICs[2]))
		
	if plot=='all':
		_plt.figure();_plt.plot(x,y)
		_plt.figure();_plt.plot(y,z)
		_plt.figure();_plt.plot(z,x)
	
		import matplotlib as _mpl
		_mpl.rcParams.update({'figure.autolayout': False})
		fig = _plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax.plot(x,y,zs=z)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')		
		ax.set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args['sigma'],args['b'],args['r'])+'\nICs = (%.3f, %.3f, %.3f)'%(ICs[0],ICs[1],ICs[2])+'\nState space')
		
		ax = fig.add_subplot(122, projection='3d')
		ax.plot(x[6::3],x[3:-3:3],zs=x[:-6:3])
		ax.set_xlabel('x(t)')
		ax.set_ylabel(r'x(t-$\tau$)')
		ax.set_zlabel(r'x(t-2$\tau$)')
		ax.set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args['sigma'],args['b'],args['r'])+'\nICs = (%.3f, %.3f, %.3f)'%(ICs[0],ICs[1],ICs[2])+'\nTime-lagged state space')
		fig.tight_layout(pad=5,w_pad=5)
		
	return ds


def predatorPrey(	N=100000,
					T=100,
					ICs={	'x0':0.9,
							'y0':0.9},
					args={	'alpha':2.0/3.0,
							'beta':4.0/3.0,
							'delta':1.0,
							'gamma':1.0},
					plot=False,
					removeMean=False,
					normalize=False):
	"""
	Lotka–Volterra (classic predator-prey problem) equations
	
	References
	----------
	* https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
	
	Examples
	--------
	Example 1::
		predatorPrey(plot=True)
	"""
	
	from scipy.integrate import solve_ivp
	
	## initialize
	dt=T/N
	
	## Solve Lorentz system of equations
	def ODEs(t,y,*args):
		X,Y = y
		alpha,beta,delta,gamma=args
		#print(X,Y,alpha,beta,delta,gamma)
		derivs=	[	alpha*X-beta*X*Y,
					delta*X*Y-gamma*Y]
		return derivs
	
	t_eval=_np.arange(0,T,dt)
	psoln = solve_ivp(	ODEs,
						[0,T],
						_np.array(list(ICs.items()))[:,1].astype(float),  # initial conditions
						args=args.values(),
						t_eval=t_eval
						)
	x,y=psoln.y
	
	## Cleanup data
	x=_xr.DataArray(	x,
						dims=['time'],
						coords={'time':t_eval},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x.time.attrs={'units':'au'}
	y=_xr.DataArray(	y,
						dims=['time'],
						coords={'time':t_eval},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	y.time.attrs={'units':'au'}
	
	if removeMean==True:
		x-=x.mean()
		y-=y.mean()
		
	if normalize==True:
		x/=x.std()
		y/=y.std()
		
	ds=_xr.Dataset({	'x':x,
						'y':y})
	
	## Optional plots
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ds.x.plot(ax=ax[0])
		ds.y.plot(ax=ax[1])
		
		fig,ax=_plt.subplots()
		ax.plot(ds.x.values,ds.y.values)
	
	return ds
	
	
def coupledHarmonicOscillator(	N=10000,
								T=1,
								ICs={	'y1_0':0,
										'x1_0':0.9,
										'y2_0':0,
										'x2_0':-1},
								args={	'k':1,
										'kappa':1,
										'm':1e-4},
								plot=False):
	"""
	Solve a coupled harmonic oscillator problem. See reference for details.
	
	Examples
	--------
	Example1 ::
		x1,x2=coupledHarmonicOscillator(plot=True,N=1e5,T=10)
	
	References
	----------
	http://users.physics.harvard.edu/~schwartz/15cFiles/Lecture3-Coupled-Oscillators.pdf

	"""
	
	import matplotlib.pyplot as _plt
	
	dt=T/N
	
	from scipy.integrate import solve_ivp
	 
	def ODEs(t,y,*args):
		y1,x1,y2,x2 = y
		k,kappa,m=args
		derivs=	[	-(k+kappa)/m*x1+kappa/m*x2,
					y1,  
 					-(k+kappa)/m*x2+kappa/m*x1,
					y2]
		return derivs

	time=_np.arange(0,T,dt)
	psoln = solve_ivp(	ODEs,
						t_span=[0,T],
						y0=_np.array(list(ICs.values())),  # initial conditions
						args=args.values(),
						t_eval=time
					)
	
	y1,x1,y2,x2 =psoln.y
	
	x1=_xr.DataArray(	x1,
						dims=['time'],
						coords={'time':time},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x1.time.attrs={'units':'au'}
	x2=_xr.DataArray(	x2,
						dims=['time'],
						coords={'time':time},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x2.time.attrs={'units':'s'}
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		x1.plot(ax=ax[0],marker='.')
		x2.plot(ax=ax[1],marker='.')
		ax[0].set_title('Coupled harmonic oscillator\n'+r'(k, kappa, m)='+'(%.3f, %.3f, %.6f)'%(args['k'],args['kappa'],args['m'])+'\nICs = (%.3f, %.3f, %.3f, %.3f)'%(ICs['y1_0'],ICs['x1_0'],ICs['y2_0'],ICs['x2_0']))

	ds=_xr.Dataset(	{'x1':x1,
					 'x2':x2})
	
	return ds


def coupledHarmonicOscillator_nonlinear(	N=500,
											dt=0.1,
											ICs={	'y1_0':0,
													'x1_0':1,
													'y2_0':0,
													'x2_0':0},
											args={	'f1':45,
													'f2':150,
													'm':1,
													'E':1e5},
											plot=False,
											verbose=False):
	"""
	Solve a coupled harmonic oscillator problem. See reference for details.
	
	Examples
	--------
	Example1 ::
		x1,x2=coupledHarmonicOscillator_nonlinear(N=10000,dt=1e-3,plot=True,verbose=True).values()
		nperseg=2000
		from johnspythonlibrary2.Process.Spectral import fft_average
		fft_average(x1,plot=True,nperseg=nperseg,noverlap=nperseg//2)
	
	References
	----------
	https://arxiv.org/pdf/1811.02973.pdf

	"""
	
	f1,f2,m,E=args.values()
	k1=(2*_np.pi*f1)**2*m
	k2=((2*_np.pi*f2)**2*m-k1)/2
# 	args={	'k1':k1,
# 			'k2':k2,
# 			'm1':m,
# 			'm2':m,
# 			'E':E}
	
	import matplotlib.pyplot as _plt
	
# 	dt
	
	from scipy.integrate import solve_ivp
	 
	def ODEs(t,y,*args):
		y1,x1,y2,x2 = y
		k1,k2,m1,m2,E=args
		if verbose:
			print(-k1*x1/m1,+k2*(x2-x1)/m1,E*x1**2/m1)
		derivs=	[	-k1*x1/m1+k2*(x2-x1)/m1+E*x1**2/m1,
					y1,  
 					-k1*x2/m2-k2*(x2-x1)/m2,
					y2]
		return derivs

	time=_np.arange(0,N)*dt
	T=time[-1]
	psoln = solve_ivp(	ODEs,
						t_span=[0,T],
						y0=_np.array(list(ICs.values())),  # initial conditions
						args=(k1,k2,m,m,E),
						t_eval=time
					)
	
	y1,x1,y2,x2 =psoln.y
	
	x1=_xr.DataArray(	x1,
						dims=['t'],
						coords={'t':time},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x1.t.attrs={'units':'au'}
	x2=_xr.DataArray(	x2,
						dims=['t'],
						coords={'t':time},
						attrs={'units':"au",
								 'standard_name': 'Amplitude'})
	x2.t.attrs={'units':'s'}
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		x1.plot(ax=ax[0],marker='.')
		x2.plot(ax=ax[1],marker='.')
		ax[0].set_title('Coupled harmonic oscillator\n'+r'(f1, f2, m, E)='+'(%.3f, %.3f, %.3f,%.3f)'%(f1,f2,m,E)+'\nICs = (%.3f, %.3f, %.3f, %.3f)'%(ICs['y1_0'],ICs['x1_0'],ICs['y2_0'],ICs['x2_0']))

	ds=_xr.Dataset(	{'x1':x1,
					 'x2':x2})
	
	return ds
