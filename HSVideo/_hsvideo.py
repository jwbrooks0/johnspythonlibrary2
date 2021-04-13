

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from scipy import fftpack as _fftpack
from scipy.signal import welch as _welch
# from scipy.signal.spectral import _spectral_helper
# from johnspythonlibrary2 import Plot as _plot
# from johnspythonlibrary2.Plot import subTitle as _subTitle, finalizeFigure as _finalizeFigure, finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Process.Misc import check_dims as _check_dims
from johnspythonlibrary2.Process.Spectral import fft as _fft
from johnspythonlibrary2.Process.Spectral import calcPhaseDifference as _calcPhaseDifference
import xarray as _xr



###############################################################################
#%% Dispersion plots

def dispersion_plot(video_data_1D, nperseg_dim1=1000,  dim2='theta', dim2_final='m', vmin=None, vmax=None, plot=True, f_units='Hz'):
	"""
	Calculates a dispersion plot from a 1D video dataset
	
	Parameters
	----------
	video_data_1D : xarray.core.dataarray.DataArray
		1D video data.  dims = ['t', spatial (e.g. theta or r)].  Time must be first.
	nperseg_dim1 : int or None
		int - Welch FFT averaging is applied to the time data where nperseg is the window size.  The output will be real.
		None - Standard FFT is applied to the time data (i.e. no windowing).  The output will be complex.
	dim2 : str
		The name of the spatial dimension
	dim2_final : str
		The name of the spatial dimension after the FFT is applied
	vmin : float
		Lower limit of the colorbar scale
	vmax : float
		Upper limit of the colorbar scale
	plot : bool
		True causes the plot to be produced.
	f_units : str
		Name of the frequency units.  (e.g. if t=t*1e3 is the input, then specify f_units='kHz'.)
		
	Returns
	-------
	X_2D : xarray.core.dataarray.DataArray
		Dipserion relationship.  Values are real if nperseg_dim1 is a number.  Complex if nperseg_dim1 is None.
	"""
	## Check dimensions
	_check_dims(video_data_1D, dims=['t',dim2])
	if video_data_1D.dims[0]!='t':
		raise Exception("The first dimension needs to be time, 't'")

	## FFT along dim2 (the spatial dimension)
	if True: 
		# preliminary steps
		dtheta = float(video_data_1D[dim2][1] -
					   video_data_1D[dim2][0]) / (2 * _np.pi)
		m = _fftpack.fftfreq(len(video_data_1D[dim2]), d=dtheta)
	
		# perform FFT
		X = _np.fft.fft(video_data_1D, axis=1)
		X = _xr.DataArray(X, dims=['t', dim2_final],
							coords=[video_data_1D['t'], m]).sortby(dim2_final)
	
		# return the results to the correct amplitude
		N = len(video_data_1D[dim2])
		X *= 1.0 / N  # use 2.0/N only if you've trimmed the negative freqs

	## FFT along time, t (dim1)
	if True:
		
		# preliminary steps
		dt = float(X.t[1] - X.t[0])
		
		# perform time-averaged (windowed) FFT if  nperseg_dim1 is a number
		if nperseg_dim1 is not None:
			freq, X_2D = _welch( 	X.data, fs=1.0/dt, nperseg=nperseg_dim1,
									noverlap=nperseg_dim1//2, return_onesided=True,
									scaling='spectrum', axis=0)
		# otherwise, perform standard fft 
		else: 
					freq = _fftpack.fftfreq(len(X['t']), d=dt)
					X_2D = _np.fft.fft(X.data, axis=0)
					N = len(video_data_1D['t'])
					X_2D *= 1.0 / N  # use 2.0/N only if you've trimmed the negative freqs

		X_2D = _xr.DataArray(X_2D, dims=['f', dim2_final],
							coords=[freq, X[dim2_final]]).sortby('f')
		X_2D.attrs={'long_name':'Spectral density','units':'au'}
		X_2D.f.attrs={'long_name':'FFT Frequency','units':f_units}
		X_2D[dim2_final].attrs={'long_name': dim2_final,'units':''}
	
	if plot==True:
		# convert to absolute value and take log10 (for vetter visualization)
		a=_np.log10(_np.abs(X_2D))
		a.attrs={'long_name':'Spectral density','units':'au, log10'}
		
		# set vmin and vmax (color scaling limits)
		if type(vmin)==type(None):
			vmin=float(a.min())
		if type(vmax)==type(None):
			vmax=float(a.max())#+0.5
		
		# plot
		fig, ax = _plt.subplots()
		a.plot(ax=ax, vmin=vmin, vmax=vmax)
		ax.set_title('dispersion plot')
	
	return X_2D
	
	
def dispersion_plot_2points(da1, da2, x_separation=1, nperseg=None, plot=True):
	# https://scholar.colorado.edu/downloads/qj72p7185
	# https://aip.scitation.org/doi/pdf/10.1063/1.2889424
	# https://aip.scitation.org/doi/pdf/10.1063/1.331279
	"""
	filename='C:\\Users\\jwbrooks\\data\\marcels_thesis_data\\20A_5sccm_5mm_6.29.2019_7.07 PM.mat'
	matData=jpl2.ReadWrite.mat_to_dict(filename)
	t=matData['t'].reshape(-1)
	da1=xr.DataArray(matData['s1'].reshape(-1), dims='t', coords=[t])
	da2=xr.DataArray(matData['s4'].reshape(-1), dims='t', coords=[t])

	x_separation=3e-3
	"""
	
	# check input
	_check_dims(da1,'t')
	_check_dims(da2,'t')
		
	# parameters
	nperseg=20000
	N_k=50
	N_f=1000
	
	# initialize arrays
	S=_np.zeros((N_k,N_f),dtype=float)
	count=_np.zeros((N_k,N_f),dtype=int)
	
	def calc_fft_and_k(x1,x2):	
		fft1=_fft(x1, plot=False).sortby('f')
		fft2=_fft(x2, plot=False).sortby('f')
		s=_np.real(0.5*(_np.conj(fft1)*fft1+_np.conj(fft2)*fft2))
		phase_diff,_,_=_calcPhaseDifference(fft1, fft2, plot=False)
		k=phase_diff/x_separation
# 		k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
# 		f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
		
		return s, k
	
	
	# calculate bin sizes
	s,k=calc_fft_and_k(da1,da2)
	k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
	f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
		
	
	# itegrate through each time window
	segs=_np.arange(0,len(da1),nperseg)
	for i,seg in enumerate(segs):
		if len(da1[seg:seg+nperseg])<nperseg:
			pass
		else:
			print(seg)
# 			
# 			fft1=fft(da1[seg:seg+nperseg], plot=False).sortby('f')
# 			fft2=fft(da2[seg:seg+nperseg], plot=False).sortby('f')
# 			s=_np.real(0.5*(_np.conj(fft1)*fft1+_np.conj(fft2)*fft2))
# 			
# 			phase_diff,_,_=calcPhaseDifference(fft1, fft2, plot=False)
# 			k=phase_diff/x_separation
# 			
# 			if i == 0:
# 				k_bins=_np.linspace(k.data.min(),k.data.max(),N_k+1)
# 				f_bins=_np.linspace(k.f.data.min(),k.f.data.max(),N_f+1)
# 				
			
			s,k=calc_fft_and_k(da1[seg:seg+nperseg], da2[seg:seg+nperseg])
			data=_pd.DataFrame()
			data['f']=s.f.data
			data['S']=s.data
			data['k']=k.data
			
			for a in range(N_k):
				for b in range(N_f):
					c=data.where((data['k']>k_bins[a])&(data['k']<k_bins[a+1])&(data['f']>f_bins[b])&(data['f']<f_bins[b+1])).dropna()
					count[a,b]+=len(c)
					S[a,b]=S[a,b]+c['S'].sum()
					
	count[count==0]=1	# prevent divide by 0 issues
	S=_xr.DataArray(S/count, dims=['k','f'],coords=[ (k_bins[1:]+k_bins[0:-1])/2, (f_bins[1:]+f_bins[0:-1])/2])
	
	if plot==True:
		fig,ax=_plt.subplots()
		count=_xr.DataArray(count, dims=['k','f'],coords=[ (k_bins[1:]+k_bins[0:-1])/2, (f_bins[1:]+f_bins[0:-1])/2])
		count.plot(ax=ax)
		
		fig,ax=_plt.subplots()
		_np.log10(S).plot(ax=ax)
		
	return S