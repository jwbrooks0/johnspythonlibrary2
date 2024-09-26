

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
# from johnspythonlibrary2.Process.Spectral import calcPhaseDifference as _calcPhaseDifference
import xarray as _xr
from scipy.stats import _binned_statistic
from scipy.optimize import minimize as _minimize


###############################################################################
# %% Reading HS video data


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

#%% binning

def _solve_for_bin_edges(numberBins=100):
	return _np.linspace(-_np.pi, _np.pi, numberBins + 1)

def create_radial_mask(video, ri=0.9, ro=1.1, fillValue=_np.nan, plot=False):
	"""
	Calculate radial mask

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	ri : float
		inner radius of mask
	ro : float
		outer radius of mask
	fillValue : int,float
		Fill value for the masked region. 0 or np.nan is standard.

	Returns
	-------
	mask : numpy.ndarray (2D)
	   Mask with 1s in the "keep" region and fillValue
	   in the "masked-out" region

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		mask=create_radial_mask(video, plot=True)
	"""

	R, _ = calc_video_polar_coordinates(video)
	mask = _np.ones(R.shape)
	mask[(R > ro) | (R < ri)] = fillValue

	if plot:
		temp = _xr.DataArray(mask, dims=['y', 'x'],
							coords=[video.y, video.x])
		fig, ax = _plt.subplots()
		temp.plot(ax=ax)

	return mask


def calc_video_polar_coordinates(video, plot=False):
	"""
	Creates polar coordinates for the video

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		calc_video_polar_coordinates(video, plot=True)

	"""

	X, Y = _np.meshgrid(video.x, video.y)
	R = _np.sqrt(X ** 2 + Y ** 2)
	Theta = _np.arctan2(Y, X)

	if plot:
		X = _xr.DataArray(X, dims=['y', 'x'], coords=[video.y, video.x])
		Y = _xr.DataArray(Y, dims=['y', 'x'], coords=[video.y, video.x])
		R_temp = _xr.DataArray(R, dims=['y', 'x'], coords=[video.y, video.x])
		Theta_temp = _xr.DataArray(Theta, dims=['y', 'x'],
								  coords=[video.y, video.x])
		fig, ax = _plt.subplots(1, 4)
		X.plot(ax=ax[0])
		ax[0].set_title('X')
		Y.plot(ax=ax[1])
		ax[1].set_title('Y')
		R_temp.plot(ax=ax[2])
		ax[2].set_title('R')
		Theta_temp.plot(ax=ax[3])
		ax[3].set_title('Theta')
		for i in range(4):
			ax[i].set_aspect('equal')

	return R, Theta

# azimuthal channel binning
def azimuthal_binning(video, numberBins, ri, ro, plot=False):
	"""
	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	numberBins : int
		Number of bins for binning.  e.g. 100
	ri : float
		Inner radius for the azimuthal binning
	ro : float
		Outer radius for the azimuthal binning
	plot : bool
		Optional plots of results

	Returns
	-------
	binned_data : xarray.core.dataarray.DataArray
		2D binned video data with coordinates in theta and time.

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		video = scale_video_amplitude(video, method='std')
		azimuthal_binning(video, 100, ri=0.9, ro=1.1, plot=True)

	"""

	# binning subfunction
	def binDataAndAverage(x, y, numberBins, plot=False):
		"""
		Bins data.

		Parameters
		----------
		x : numpy.ndarray
			independent variable
		y : numpy.ndarray
			dependent variable
		numberBins : int
			number of bins
		plot : bool
			Optional plot of results

		Returns
		-------
		xarray.core.dataarray.DataArray
			DataArray containing the binned results
		Example
		-------
		Example 1::

			x = np.linspace(0, 2 * np.pi, 1000) - np.pi
			y = np.cos(x) + 1 * (np.random.rand(x.shape[0]) - 0.5)
			numberBins = 100
			bin_results = binDataAndAverage(x, y, numberBins, plot=True)

		"""
		bin_edges = _solve_for_bin_edges(numberBins)

		# bin y(x) into discrete bins and average the values within each
		y_binned, _, _ = _binned_statistic(x, y, bins=bin_edges,
										  statistic='mean')
		x_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

		if plot:
			da_raw = _xr.DataArray(y, dims=['x'], coords=[x]).sortby('x')
			fig, ax = _plt.subplots()
			da_raw.plot(ax=ax, label='raw data')
			ax.plot(x_bins, y_binned, label='binned data',
					marker='s', ms=3, linestyle='--')
			ax.legend()

		return _xr.DataArray(y_binned, dims='Theta', coords=[x_bins])

	# create radial mask
	R, Theta = calc_video_polar_coordinates(video)
	mask = create_radial_mask(video, ri=ri, ro=ro)

	# bin and average each time step in the data
	binned_data = _np.zeros((video.t.shape[0], numberBins))
	for i, t in enumerate(video.t.data):
		unbinned_data = _pd.DataFrame()
		unbinned_data['theta'] = Theta.reshape(-1)
		unbinned_data['radius'] = R.reshape(-1)
		unbinned_data['data'] = (video.sel(t=t).data * mask).reshape(-1)
		unbinned_data = unbinned_data.dropna()

		if i == 0 and plot:
			plot2 = True
		else:
			plot2 = False
			
		if i==0:
			print('Average number of pixels per bin:',unbinned_data.shape[0]/numberBins)


		out = binDataAndAverage(unbinned_data.theta.values,
								unbinned_data.data.values,
								numberBins, plot=plot2)
		
		if i == 0:
			number_of_NaNs = _np.isnan(out).sum()
			if number_of_NaNs > 0:
				print('NaNs encounted in binning: ', number_of_NaNs)
		binned_data[i, :] = out

	binned_data = _xr.DataArray(binned_data, dims=['t', 'theta'],
							   coords=[video.t.data.copy(), out.Theta])

	if plot:
		fig, ax = _plt.subplots()
		binned_data.plot(ax=ax)

	return binned_data


#%% Circular/annulus detection


def _circle(ax, xy=(0, 0), r=1, color='r', linestyle='-',
		   alpha=1, fill=False, label=''):
	"""
	Draws a circle on an AxesSubplot (ax) at origin=(xy) and radius=r
	"""
	circle1 = _plt.Circle(xy, r, color=color, alpha=alpha,
						 fill=fill, linestyle=linestyle)
	ax.add_artist(circle1)
	

def scale_video_spatial_gaussian(video, guess=[], plot=False, verbose=False):
	"""
	Scale (center and normalize) the video's cartesian coordinates
	using an annular Gaussian fit

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
	   the video
	guess : list (empty or of 6 floats)
		Guess values for the fit.
		Default is an empty list, and a "reasonable" guess is used.
		[amplitude, channel x center, channel y center,
		channel radius, channel width, offset]
	plot : bool
		optional plot of the results
	verbose : bool
		optionally prints misc steps of the fit

	Returns
	-------
	video : xarray.core.dataarray.DataArray
		the video with coordinates scaled
	fit_params : dict
		Fit parameters

	Examples
	--------
	Example 1 ::

		video = create_fake_video_data()
		video_scaled, params = scale_video_spatial_gaussian(video, plot=True,
															verbose=True)
	"""

	# convert video to time averaged image
	image = calc_video_time_average(video.copy())

	# create Cartesian grid
	X, Y = _np.meshgrid(image.x.data, image.y.data)

	# annular Gaussian model, assumed form of the channel
	def model(image, params):
		a0, x0, y0, r0, sigma0, offset = params

		def gaussian(a, r, sigma, R):
			return a * _np.exp(-0.5 * ((R - r) / sigma) ** 2)

		R0 = _np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
		Z = gaussian(a0, r0, sigma0, R0) ** 1 + offset

		return Z

	# Generate a reasonable guess and guess image
	if len(guess) < 6:
		sh = image.shape
		guess = [1, sh[1] // 2, sh[0] // 2, _np.min(sh) / 3, _np.min(sh) / 4, 4]

	# Function that minimizes (i.e. fits) the parameters to the model
	def min_func(params):
		Z = model(image.data, params)
		error = _np.abs((image.data - Z)).sum()
		if verbose:
			print('error = %.6f' % error)
		return error

	# perform fit
	fit = _minimize(min_func, guess)
	a0, x0, y0, r0, sigma0, offset = fit.x
	fit_params = {'a0': a0, 'x0': x0, 'y0': y0, 'r0': r0,
				  'sigma0': sigma0, 'offset': offset}

	# optional plot of results
	if plot:
		Z_fit = _xr.DataArray(model(image, fit.x),
							 dims=image.dims, coords=image.coords)
		Z_guess = _xr.DataArray(model(image, guess),
							   dims=image.dims, coords=image.coords)

		fig, ax = _plt.subplots(1, 2, sharey=True)
		image.sel(x=x0, method='nearest').plot(ax=ax[0], label='data',
											   color='k')
		Z_fit.sel(x=x0, method='nearest').plot(ax=ax[0], label='fit',
											   linestyle='--',
											   color='tab:blue')
		ax[0].set_title('x=x0=%.1f' % x0)
		image.sel(y=y0, method='nearest').plot(ax=ax[1], label='data',
											   color='k')
		Z_fit.sel(y=y0, method='nearest').plot(ax=ax[1], label='fit',
											   linestyle='--',
											   color='tab:blue')
		ax[1].set_title('y=y0=%.1f' % y0)
		ax[0].legend()
		ax[1].legend()

		image['x'] = (image.x - x0) / r0
		image['y'] = (image.y - y0) / r0

		fig0, ax0 = _plt.subplots(1, 4)

		ax0[0].imshow(image, origin='lower')
		ax0[0].set_title('actual')

		ax0[1].imshow(Z_guess, origin='lower')
		ax0[1].set_title('guess')

		ax0[2].imshow(Z_fit, origin='lower')
		ax0[2].set_title('fit')

		ax0[3].imshow(image, origin='lower')
		ax0[3].set_title('actual with fit')

		_circle(ax0[3], xy=(x0, y0), r=r0, fill=False, linestyle='--')
		_circle(ax0[3], xy=(x0, y0), r=r0 + sigma0 * 1.5, fill=False)
		_circle(ax0[3], xy=(x0, y0), r=r0 - sigma0 * 1.5, fill=False)

	# apply correction to the video
	video = video.copy()
	video['x'] = (video.x - x0) / r0
	video['y'] = (video.y - y0) / r0

	return video, fit_params


#%% Video processing, misc

def calc_video_time_average(video, plot=False):
	"""
	calculate time averaged image

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		mask = calc_video_time_average(video, plot=True)
	"""
	ave = video.mean(dim='t')
	if plot:
		fig, ax = _plt.subplots()
		ave.plot(ax=ax)
		ax.set_title('time average')
	return ave

