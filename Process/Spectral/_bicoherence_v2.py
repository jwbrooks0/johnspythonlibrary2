

# %% Import standard libraries
import numpy as _np
import matplotlib.pyplot as _plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
import xarray as _xr
try:
	from scipy.signal.spectral import _spectral_helper
except ImportError:
	from scipy.signal._spectral_py import _spectral_helper


# %% Import custom libraries
# from johnspythonlibrary2.Plot import finalizeFigure as _finalizeFigure #, finalizeSubplot as _finalizeSubplot, subTitle as _subTitle, 
from johnspythonlibrary2.Process.Spectral import fft_average as _fft_average, signal_spectral_properties as _signal_spectral_properties
# from johnspythonlibrary2.Process.Misc import check_dims # subtract_mean_and_normalize_by_std, 



# %% plotting

def _finalizeFigure(fig,
				   figSize=[],
				   **kwargs):
	""" 
	Performs many of the "same old" commands that need to be performed for
	each figure but wraps it up into one function
	
	Parameters
	----------
	fig : matplotlib.figure.Figure
		Figure to be modified
	figSize : list of floats
		Figure size.  Units in inches.  E.g. figSize=[6,4]
	"""
	_FONTSIZE = 8
	
	# default input parameters
	params={	'title': '',
				'h_pad' : 0.25,
				'w_pad' : 0.25,
				'fontSizeTitle': _FONTSIZE+2,
				'pad': 0.5,
				}
	
	# update default input parameters with kwargs
	params.update(kwargs)
	
	# fig.suptitle(title) # note: suptitle is not compatible with set_tight_layout
	
	if params['title'] != '':
		fig.suptitle(params['title'], fontsize=params['fontSizeTitle'])
		
	if figSize != []:
		fig.set_size_inches(figSize)
		
	# fig.set_tight_layout(True)
	fig.tight_layout(h_pad=params['h_pad'], w_pad=params['w_pad'], pad=params['pad']) # sets tight_layout and sets the padding between subplots
			

def _subTitle(	ax,
				string,
				xy=(0.5, 0.98),
				box=True,
				textColor='k',
				xycoords='axes fraction',
				fontSize=8,
				horizontalalignment='center',
				verticalalignment='top'):
	"""
	wrapper for the annotate axis function.  the default setting is for a
	figure subtitle at the top of a particular axis
	
	Parameters
	----------
	ax : matplotlib.axes._subplots.AxesSubplot
		Axis that will receive the text box
	string : str
		String to put in textbox
	xy : tuple
		(x,y) coordinates for the text box
	box : bool
		True - Creates a box around the text
		False - No box
	textColor : str
		text color
	xycoords : str
		type of coordinates.  default = 'axes fraction'
	fontSize : int
		text font size
	horizontalalignment : str
		'center' - coordinates are cenetered at the center of the box
		'left'
		'right'
	verticalalignment : str
		'top' - coordinates are centered at the top of the box
	
	"""
	if box is True:
		box = dict(boxstyle="square, pad=.25", fc="w", edgecolor='k')
	else:
		box=None

	ax.annotate(string, 
				xy=xy, 
				color=textColor,
				xycoords=xycoords, 
				fontsize=fontSize,
				horizontalalignment=horizontalalignment, 
				verticalalignment=verticalalignment,
				bbox=box)
	

def bicoherence_plot(da, 
					 bicoh, 
					 nperseg, 
					 vmin=None, 
					 vmax=None, 
					 title=None, 
					 drawRedLines=[], 
					 hz_to_kHz=False):

	if vmin is None:
		vmin = float(_np.abs(bicoh).min())
	if vmax is None:
		vmax = float(_np.abs(bicoh).max())
		
	if hz_to_kHz is True:
		da['t'] = da['t'] * 1e3
		f1 = bicoh.f1
		f2 = bicoh.f2
		
		f_units = 'kHz'
		f1 = f1 * 1e-3
		f2 = f2 * 1e-3
		f1.attrs = {'units': f_units, 'long_name': r'$f_1$'}
		f2.attrs = {'units': f_units, 'long_name': r'$f_2$'}
		bicoh['f1'] = f1
		bicoh['f2'] = f2
	else:
		f_units = 'Hz'

	# initialize figure
	fig = _plt.figure()
	
	# subplot 1
	ax1 = _plt.subplot(111)
	ax1.set_aspect('equal')
	levels = _np.linspace(vmin, vmax, 20 + 1)
	im = _np.abs(bicoh).plot(ax=ax1, levels=levels, add_colorbar=False, vmin=vmin, vmax=vmax )

	# trick to get the subplots to line up correctly
	divider = _make_axes_locatable(ax1)
	ax2 = divider.append_axes("bottom", size="50%", pad=0.5, sharex=ax1)
	cax = divider.append_axes("right", size="5%", pad=0.08)
	cbar = _plt.colorbar(im, ax=ax1, cax=cax, ticks= _np.linspace(vmin, vmax, 6))
	cbar.set_label('$b^2$')
	
	# subplot 2
	fft_results = _fft_average(	da,
								nperseg=nperseg,
								noverlap=0,
								trimNegFreqs=False,
								sortFreqIndex=True,
								f_units=f_units,
								realAmplitudeUnits=True)
	# fft_results/=fft_results.sum() # normalize
	_np.abs(fft_results).plot(ax=ax2, yscale='log')
	ax2.set_ylabel('FFT mag. (au)')
	
	# optional, draw red lines
	for y0 in drawRedLines:
		f1 = bicoh.coords['f1'].data
		f2 = bicoh.coords['f2'].data
		ax1.plot(f1, y0-f1, color='r', linestyle='--', linewidth=0.5)

	# finalize
	_subTitle(ax1, 'Bicoherence', xy=(0.98, .98), horizontalalignment='right',)
	_subTitle(ax2, 'FFT', xy=(0.98, .98), horizontalalignment='right')
	fig.set_size_inches(6, 4)
	ax1.set_title(title)
	_finalizeFigure(fig)
	
	return fig

# %% bicoherence subfunctions


def _bicoherence_helper(stft, f_units='Hz', verbose=False, normalize=True):
	""" 
	subfunction for calculating bicoherence from a signal's STFT
	
	Parameters
	----------
	stft : xarray.DataArray
		This is the stft results just prior to calculating the bicoherence
	f_units : str
		Units for frequency
	verbose : bool
		Optional print
	normalize : bool
		Optionally normalize the bicoherence results.  Default is True which provides the traditional bicoherence results.
	"""

	# calculate bicoherence numerator and denominators
	for i, ti in enumerate(stft.t.data): # TODO(John) Figure out how to vectorize this step.  It runs very slow...
		if verbose is True:
			print(i, ti)
			
		b, FiFj, conjFij = _bispectrum(stft.sel(t=ti), returnAll=True, plot=False, f_units=f_units)
		
		if i == 0:
			numerator = FiFj * conjFij
			denom1 = _np.abs(FiFj)**2
			denom2 = _np.abs(conjFij)**2
		else:
			numerator += FiFj * conjFij
			denom1 += _np.abs(FiFj)**2
			denom2 += _np.abs(conjFij)**2
			
	# combine numerator and denominator
	if normalize is True:
		bicoh = numerator**2 / (denom1.data * denom2.data)
	else:
		bicoh = numerator**2
	
	return bicoh


def _bispectrum(	da,
					firstQuadrantOnly=False,
					plot=False,
					returnAll=False,
					f_units='Hz'):
	"""
	Calculate the bispectrum for each STFT window
	
	Parameters
	----------
	da : xarray.DataArray
		Single time window of the STFT
		
	Returns
	-------
	b : xarray.DataArray
		Bispectrum results
	(optional) m1 : 
		m1 = F(f1) * F(f2)
	(optional) m2 :
		m2 = conj(F(f1+f2))	
	
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
		b=_bispectrum(da,plot=True,firstQuadrantOnly=True)
		
		y=0.5*np.cos(2*np.pi*t*(f2+f1)+phase1)+0.5*np.cos(2*np.pi*t*(f2-f1)+phase2)+noise
		da=xr.DataArray(y,
					  dims=['t'],
					  coords={'t':t})
		da=fft(da,plot=True).sortby('f')
		b=_bispectrum(da,plot=True,firstQuadrantOnly=True)
		
	"""
	
	N = da.shape[0]
	signal = da.data

	# convert indices to integers (makes it easier to index later on)
	index = da.f.copy().data
	df = index[1] - index[0]
	index = _np.round(index / df).astype(int)
	
	# Solve for M1 = F(f1) * F(f2)
	def M1(signal, index, index1name='f2', index2name='f1'):
		return _xr.DataArray(	_np.outer(signal, signal),
				 			    dims=[index1name, index2name],
								coords={index1name:index, index2name:index})
	
	# Solve for M2 = F(f1+f2)
	def M2(signal, index, index1name='f2', index2name='f1'):
		x = signal
		f = index
		
		# padding signal and index to account for the extremes of min(index)+min(index) and max(index)+max(index)
		f_long = _np.arange(f.min() * 2, 2 * (f.max() + (f[1] - f[0])), f[1] - f[0])
		x_long = _np.concatenate(	(_np.zeros(N // 2) * _np.nan,
								  x,
								  _np.zeros(int(_np.ceil(N / 2))) * _np.nan))
		x_long = _xr.DataArray(	x_long,
				 			    dims=['f'],
								coords={'f': f_long})
		
		# calculate each permutation of f1+f2
		f1temp, f2temp = _np.meshgrid(f, f)
		fsumlist = (f1temp + f2temp).reshape(-1)
		
		# find M2 = F(f1+f2)
		return _xr.DataArray(	x_long.loc[fsumlist].data.reshape(N, N),
				 			    dims=[index1name, index2name],
								coords={index1name: f, index2name: f})
	
	# bispectrum
	m1 = M1(signal=signal, index=index)
	m2 = _np.conj(M2(signal, index))
	f1 = m1.f1.data * df
	f1 = _xr.DataArray(f1, dims='f1', coords=[f1], attrs={'units': f_units, 'long_name': r'$f_1$'})
	f2 = m1.f2.data * df
	f2 = _xr.DataArray(f2, dims='f2', coords=[f2], attrs={'units': f_units, 'long_name': r'$f_2$'})
	m1['f1'] = m1.f1.data * df
	m1['f2'] = m1.f2.data * df
	m2['f1'] = m2.f1.data * df
	m2['f2'] = m2.f2.data * df
	b = m1 * m2
	
	if firstQuadrantOnly is True:
		b = b[b.f2 >= 0, b.f1 >= 0]
	
	if plot is True:
		_plt.figure()
		_np.abs(b).plot()
		
	if returnAll is False:
		return b
	else:
		return b, m1, m2


#%% main function(s)

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
					precondition_signal=True,
					vmin=None,
					vmax=None,
					normalize=True):
	"""
	Bicoherence and bispectrum analysis.  This algorithm is based on [Kim1979].
	
	This code appears to be correct. 
	
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
						windowFunc='hann',
						mask='A',
						plot=True)
		_plt.gcf().savefig('figure1.png')
		
		### Figure 2
		thetad=randomPhase(M,seed=3)
		x2=(baseSignal+0.5*sigGen(t,fd,thetad)).flatten()
		da=xr.DataArray(x2,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='hann',
						mask='A',
						plot=True)
		_plt.gcf().savefig('figure2.png')
		
		### Figure 3
		x3=(baseSignal+0.5*sigGen(t,fd,thetab+thetac)).flatten()
		da=xr.DataArray(x3,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='hann',
						plot=True,
						mask='A',
# 						drawRedLines=[fb,fc,fd],
						drawRedLines=[fd])
		_plt.gcf().savefig('figure3.png')
		
		### Figure 4
		x4=(baseSignal+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
		da=xr.DataArray(x4,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
						windowFunc='hann',
						mask='A',
						plot=True,
# 						drawRedLines=[fb,fc,fd,fa],
						drawRedLines=[fc,fd])
		_plt.gcf().savefig('figure4.png')
		
		### Figure 5
		x5=(baseSignal+0.5*sigGen(t,fd,thetad)+1*sigGen(t,fb,thetab)*sigGen(t,fc,thetac)).flatten()
		da=xr.DataArray(x5,dims=['t'],coords={'t':t})
		dfBicoh=bicoherence(	da,
						nperseg=recordLength,
									windowFunc='hann',
									mask='A',
									plot=True,
									drawRedLines=[fc,fd])
		_plt.gcf().savefig('figure5.png')


	Example 2 ::
		
        print("work in progress")
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

	"""
	
	if 'time' in da.dims:
		da = da.rename({'time': 't'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found' % (str(da.dims)))
				
	if precondition_signal==True:
		da = (da.copy() - da.mean(dim='t').data) / da.std(dim='t').data

	dt, fsamp, fn, _, _ = _signal_spectral_properties(da, nperseg=nperseg, verbose=verbose).values()
	
	# Solve for STFT 
	print(windowFunc)
	f, t, stft_results = _spectral_helper(	da.data,
											da.data,
											fs = 1 / (da.t.data[1] - da.t.data[0]),
											window=windowFunc,
											nperseg=nperseg,
											noverlap=0,
											return_onesided=False,
											mode='stft')
# 	stft_results = _pd.DataFrame(stft_results, index=f, columns=t) #TODO remove pandas step
# 	stft_results.index.name = 'f'
# 	stft_results.columns.name = 't'
	f = _xr.DataArray(f, dims='f', coords=[f], attrs = {'units': f_units})
	if 'units' in list(da.t.attrs.keys()):
		t_units = da.t.attrs['units']
	else:
		t_units = ''
	t = _xr.DataArray(t, dims='t', coords=[t], attrs = {'units': t_units})
	stft_results = _xr.DataArray(stft_results, dims=['f', 't'], coords=[f, t]).sortby('f')
	
	# calculate bicoherence
	bicoh = _bicoherence_helper(stft_results, f_units=f_units, normalize=normalize)
	
	# (optional) apply mask
	if mask == 'AB':
		f1 = bicoh.coords['f1']
		f2 = bicoh.coords['f2']
		
		a = (f1 <= f2) & (f1 >= -f2)
		b = (a * 1.0).values
		b[b == 0] = _np.nan
		bicoh *= b
		#bicoh=bicoh[bicoh.f2>=0,:]
		bicoh = bicoh[:, bicoh.f1 >= 0]
		bicoh = bicoh[bicoh.f2 <= fn / 2]
	elif mask == 'A':
		f1 = bicoh.coords['f1']
		f2 = bicoh.coords['f2']
		
		a = (f2 <= f1) & (f2 >= -f1)
		b = (a * 1.0).values
		b[b == 0] = _np.nan
		bicoh *= b
		bicoh = bicoh[bicoh.f2 >= 0, :]
		bicoh = bicoh[:, bicoh.f1 >= 0]
		bicoh = bicoh[bicoh.f2 <= fn / 2]
	elif mask == 'none' or mask == 'None':
		pass
	else:
		raise Exception('Improper mask value encountered : %s' % (str(mask)))
		
	# finalize xarray
	bicoh.f1.attrs = {'units': f_units, 'long_name': r'$f_1$'}
	bicoh.f2.attrs = {'units': f_units, 'long_name': r'$f_2$'}
	
	# (optional) plot
	if plot is True:
		bicoherence_plot(da, bicoh, nperseg, vmin, vmax)
			
	return bicoh


# %% Functions under development

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
# 	import numpy as np
# 	from mpl_toolkits.axes_grid1 import make_axes_locatable
# 	from scipy.signal.spectral import _spectral_helper
# 	import pandas as pd
# 	import xarray as xr
# 	import matplotlib.pyplot as plt
	
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'%(str(da.dims)))
		
	if precondition_signal==True:
		da=(da.copy() - da.mean(dim='t').data) / da.std(dim='t').data

	params = _signal_spectral_properties(da, nperseg=nperseg, verbose=verbose)
	
	# step 2 : generate bicoherence from actual data
	b=bicoherence(da, nperseg=nperseg, plot=plot, mask='None')
	
	
	
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
	stft_results_abs = _np.abs(stft_results)
	
	# step 3a: Generate random phase for each fourier component in each time window.  Do this R times.
	theta = _np.random.rand(stft_results_abs.shape[0], stft_results_abs.shape[1], R)

	# step 3b: Simulate (monte carlo) fourier data using real amplitude and random phase
	stft_results_mc = _np.repeat(stft_results_abs[:, :, _np.newaxis], R, axis=2) * _np.exp(1j * theta)
	stft_results_mc = _xr.DataArray(stft_results_mc, dims=['f', 't', 'r'], coords=[f, t, _np.arange(R)]).sortby('f')

	# step 3c: calculate bicoh for each r in range(R) to create the probability distribution function for each pair of frequency coordinates
	if True:  #TODO this is painfully slow.  figure out how to vectorize
		mc_results = _xr.DataArray( _np.zeros((stft_results_mc.f.shape[0], stft_results_mc.f.shape[0], R),dtype=complex))
		for i, ri in enumerate(range(R)):
			print('%d/%d'%(ri + 1 ,R), end=' ')
			mc_results[:, :, i]=_bicoherence_helper(stft_results_mc[:, :, i])
		print('')
		mc_results = _np.abs(mc_results)
	else:
		print('work in progress')
		# look into Numba functions 
		
	# step 4: alpha is already defined
	
	# step 5: Calculate critical bicoherence value (noise floor)
	from scipy.interpolate import interp1d as _interp1d
	def integrate_along_prob_function_integral_equals_value(x,y,alpha=0.9):
		# y(x) is the probability function
		y /= y.sum() # normalize such that integral is equal to 1
		p = _np.cumsum(y) # p = cumalative sum of y(x)
		f = _interp1d(p,x) # create interpolation of function, x(p)
		return float(f(alpha)) # find where x(p) = alpha
			
	def hist_1d(a, alpha=0.9):
		if _np.isnan(a).sum() > 0:
			return float(0)
		else:
			hist, bin_edges = _np.histogram(a)
			hist = hist.astype(float)/hist.sum()
			bin_centers=(bin_edges[1:] + bin_edges[:-1]) / 2
			a = integrate_along_prob_function_integral_equals_value(x=bin_centers, y=hist, alpha=alpha)
			return a

	counts = _np.apply_along_axis(hist_1d, axis=2, arr=mc_results.data, alpha=alpha)
	counts = _xr.DataArray(	counts,
							dims=b.dims,
							coords=b.coords )
	
	if plot is True:
		fig, ax = _plt.subplots()
		counts.plot(ax=ax)
		ax.set_title('Critical bicoherence values')
	
	# Steps 6 and 7: Any place where bicoherence values are less than the critical bicoh values, set equal to NaN.  Otherwise, keep. 
	b_abs = _np.abs(b).data
	b_abs[b_abs < counts.data] = _np.nan
	b_final = _xr.DataArray(b_abs, dims=b.dims, coords=b.coords)
	
	if plot is True:
		fig, ax = _plt.subplots()
		b_final.plot(ax=ax)
	
	return b_final
# 	counts= counts.


def bicoherence_2D(	da, 
					nperseg=500, 
					verbose=True, 
					precondition_signal=False, 
					windowFunc='hann', 
					m_range=_np.arange(-5,5.1,1, 
					dtype=int)
					):
	"""
	
	References
	----------
	 * See Appendix A in https://aip.scitation.org/doi/10.1063/1.3429674
	 
	"""
	print("work in progress")
	
	
	
	
		
# 	import numpy as np
# 	from mpl_toolkits.axes_grid1 import make_axes_locatable
# 	import pandas as pd
# 	import xarray as xr
# 	import matplotlib.pyplot as plt
# 	from johnspythonlibrary2.Process.Misc import check_dims, subtract_mean_and_normalize_by_std
# 	from johnspythonlibrary2.Process.Spectral import signal_spectral_properties
	
	# make sure 't' and 'theta' are in dimensions
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	for dim in ['t','theta']:
		if dim not in da.dims:
			raise Exception('Dimension, %s, not present.  Instead, %s found' % (dim, str(da.dims)))
	# check_dims(da, dims=['t','theta'])
	
	# optional precondition signal
	if precondition_signal is True:
		dim = 't'
		da = (da.copy() - da.mean(dim=dim).data) / da.std(dim=dim).data
		# da = subtract_mean_and_normalize_by_std(da,'t')

	dt, fsamp, fn, _, _ = _signal_spectral_properties(da, nperseg=nperseg, verbose=verbose).values()
	
	# Solve for the STFT results from each time window
	f,t,stft_results = _spectral_helper(	da.data,
											da.data,
											axis=0,
											fs=1 / (da.t.data[1] - da.t.data[0]),
											window=windowFunc,
											nperseg=nperseg,
											noverlap=0,
											return_onesided=False,
											mode='stft')
	da2 = _xr.DataArray(stft_results, dims=['f', da.dims[1], 't'], coords=[f, da.coords['theta'].data, t]).sortby('f')
	
	# Solve for FFT (STFT with 1 window) of azimuthal
	dtheta = da2.coords['theta'].data[1] - da2.coords['theta'].data[0]
	m, t0 ,stft_results = _spectral_helper(	da2.data,
											da2.data,
											axis=1,
											fs=1 / dtheta,
											window=windowFunc,
											nperseg=da2.shape[1],
											noverlap=0,
											return_onesided=False,
											mode='stft')
	m = _np.round(m * _np.pi * 2).astype(int)
	X2D = _xr.DataArray(stft_results[:,:,:,0], dims=['f','m','t'], coords=[f,m,t]).sortby('m')
# 	N=X2D.shape[0]
	
# 	signal_1 = X2D.
	
	def bispectrum_2D(signal_1, signal_2, signal_3, f_units='Hz'):
		
		f_old = signal_1.f.data
		df = f_old[1] - f_old[0]
		f_new = _np.round(f_old / df).astype(int)
		
		# Solve for M1 = F(f1)*F(f2)
		def M1():
			return _xr.DataArray(	_np.outer(signal_1, signal_2),
					 			    dims=['f2', 'f1'],
									coords={'f2':f_old, 'f1':f_old})
		
		# Solve for M2 = F(f1+f2)
		def M2():

			f = f_new
			x = signal_3.data
			N = X2D.shape[0]
			
			# padding signal and index to account for the extremes of min(index)+min(index) and max(index)+max(index)
			f_long = _np.arange(f.min() * 2, 2 * (f.max() + (f[1] - f[0])), f[1] - f[0])
			x_long = _np.concatenate(	(_np.zeros(N // 2) * _np.nan,
									  x,
									  _np.zeros(int(_np.ceil(N / 2))) * _np.nan))
			x_long = _xr.DataArray(	x_long,
					 			    dims=['f'],
									coords={'f': f_long})
			
			# calculate each permutation of f1+f2
			f1temp,f2temp = _np.meshgrid(f, f)
			fsumlist = (f1temp + f2temp).reshape(-1)
			
			# find M2 = F(f1+f2)
			return _xr.DataArray(	x_long.loc[fsumlist].data.reshape(N, N),
					 			    dims=['f2', 'f1'],
									coords={'f2':f_old, 'f1':f_old})
		
		# bispectrum
		m1 = M1()
		m2 = _np.conj(M2())
		b = m1 * m2
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
		m3 = m1 + m2
		
		if m3 in X2D.m.data:
			
			for i, ti in enumerate(X2D.t.data): # TODO(John) Figure out how to vectorize this step
				print(i, ti)
				signal_1 = X2D.sel(t=ti, m=m1)
				signal_2 = X2D.sel(t=ti, m=m2)
				signal_3 = X2D.sel(t=ti, m=m3)
				b, FiFj, conjFij=bispectrum_2D(signal_1, signal_2, signal_3)
				
				if i == 0:
					numerator = FiFj * conjFij
					denom1 = _np.abs(FiFj)**2
					denom2 = _np.abs(conjFij)**2
				else:
					numerator += FiFj * conjFij
					denom1 += _np.abs(FiFj)**2
					denom2 += _np.abs(conjFij)**2
					
			# finish bicoherence calc
			bicoh = numerator**2 / (denom1.data * denom2.data)
			
			return bicoh.sortby('f1').sortby('f2')
		
		else:
			print('invalid m3 value')
			return []
		
	def _bicoherence_2D_helper1(X2D, m_range):
		
		m1_range, m2_range = _np.meshgrid(m_range, m_range)
		m1_range = m1_range.reshape(-1)
		m2_range = m2_range.reshape(-1)
		
		b = _bicoherence_2D_helper2(X2D, 0, 0)
		
		data = _xr.DataArray(_np.zeros((b.shape[0], b.shape[1], len(m_range), len(m_range)), dtype=complex),
							  dims=['f2', 'f1', 'm1', 'm2'],
							  coords=[b.f2, b.f1, m_range.astype(int), m_range.astype(int)])
		for m1, m2 in zip(m1_range, m2_range):
			#m3=m1+m2
			#print(m1,m2,m3)
			
			b = _bicoherence_2D_helper2(X2D, m1, m2)
			if len(b) > 0:
				data.loc[:, :, m1, m2] = b
			else:
				data.loc[:, :, m1, m2] = _np.nan
				
		return data
				
# 	m_range=np.arange(-4,4)
	result=_bicoherence_2D_helper1(X2D, m_range)
	
	return result

	
# %% Examples


	
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
			

	
def example_stationary_waves_Kim_1979():
	""" 
	This function approximately reproduces the example in Kim's 1979 paper
	
	References
	---------- 
	 * Y.C. Kim and E.J. Powers, IEEE Transactions on Plasma Science 7, 120 (1979).
	"""
	
	import matplotlib.pyplot as plt
	import numpy as np
	import xarray as xr
	
	plt.close('all')
	
	### subfunctions
	def randomPhase(n=1,seed=0):
		np.random.seed(seed)
		return (np.random.rand(n)-0.5)*np.pi*2
	
	def sigGen(t,f,theta):
		M=len(theta)
		N=len(t)//M
		T,Theta=np.meshgrid(t[0:N],theta)
		return 1*np.cos(2*np.pi*T*f+Theta)
	
	### initialize
	mask = 'A'
	numberRecords = 64
	recordLength = 128 * 4
	N = recordLength
	M = numberRecords
	dt = 5e-1
	t = np.arange(0, N * M) * dt
	fN = 1
	fa = 0.220 * fN
	fb = 0.375 * fN
	nperseg = recordLength
	
	thetaa = randomPhase(M, seed=1)
	thetab = randomPhase(M, seed=2)
	
	np.random.seed(0)
	noise = np.random.normal(0, 0.1, (M, N))
	
	baseSignal = sigGen(t, fa, thetaa) + sigGen(t, fb, thetab) + noise

	
	### Figure 1
	x1 = (baseSignal).flatten()
	da = xr.DataArray(x1, dims=['t'], coords={'t':t})
	dfBicoh = bicoherence(	da,
					nperseg=nperseg,
					windowFunc='hann',
					mask=mask,
					plot=True,
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)$',
					vmin=0,
					vmax=1)
	_plt.gcf().savefig('images/figure1.png')
	
	### Figure 2
	thetac = randomPhase(M,seed=3)
	x2 = (baseSignal +0.5 * sigGen(t, fa + fb, thetac)).flatten()
	da = xr.DataArray(x2, dims=['t'], coords={'t': t})
	dfBicoh = bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					mask=mask,
					plot=True,
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+\frac{1}{2}cos(2 \pi (f_a+f_b) t + \theta_c)$',
					vmin=0,
					vmax=1)
	_plt.gcf().savefig('images/figure2.png')
	
	### Figure 3
	x3 = (baseSignal + 0.5 * sigGen(t, fa + fb, thetaa + thetab)).flatten()
	da = xr.DataArray(x3, dims=['t'], coords={'t': t})
	dfBicoh = bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					plot=True,
					mask=mask,
# 						drawRedLines=[fb,fc,fd],
					drawRedLines=[fa+fb],
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+\frac{1}{2}cos(2 \pi (f_a+f_b) t + \theta_a+\theta_b)$',
					vmin=0,
					vmax=1)
	_plt.gcf().savefig('images/figure3.png')
	
	### Figure 4
	x4 = (baseSignal + 1 * sigGen(t, fa, thetaa) * sigGen(t, fb, thetab)).flatten()
	da = xr.DataArray(x4, dims=['t'], coords={'t': t})
	dfBicoh = bicoherence(	da,
					nperseg=recordLength,
					windowFunc='hann',
					mask=mask,
					plot=True,
# 						drawRedLines=[fb,fc,fd,fa],
					drawRedLines=[fa+fb,fb],
					fft_scale='linear',
					title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+cos(2 \pi f_a t + \theta_a)cos(2 \pi f_b t + \theta_b)$',
					vmin=0,
					vmax=1)
	_plt.gcf().savefig('images/figure4.png')
	
	### Figure 5
	x5 = (baseSignal + 0.5 * sigGen(t, fa + fb, thetac) + 1 * sigGen(t, fa, thetaa) * sigGen(t, fb, thetab)).flatten()
	da = xr.DataArray(x5, dims=['t'], coords={'t': t})
	dfBicoh = bicoherence(	da,
							nperseg=recordLength,
							windowFunc='hann',
							mask=mask,
							plot=True,
							drawRedLines=[fa+fb,fb],
							fft_scale='linear',
							title=r'$y(t)=cos(2 \pi f_a t + \theta_a)+cos(2 \pi f_b t + \theta_b)+cos(2 \pi (f_a+f_b) t + \theta_c)+cos(2 \pi f_a t + \theta_a)cos(2 \pi f_b t + \theta_b)$',
							vmin=0,
							vmax=1)
	_plt.gcf().savefig('images/figure5.png')

#%% main
# if __name__ == '__main__':
# 	example1(mask='none',fft_scale='linear')
