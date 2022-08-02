# -*- coding: utf-8 -*-
"""
The file contains the majority of the functions used in the HSI analysis 
discussed in the journal article.  The goal of this file is to be shared with
other researchers to help with their work.  My only request is that we please 
be given credit in any publication whenever this toolkit is used.

@author: John "Jack" Brooks (jwbrooks0@gmail.com, john.brooks.ctr@nrl.navy.mil)
and Alan Kaptanoglu (akaptano@uw.edu)
"""

##############################################################################
# %% Load libraries

# 3rd party libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fftfreq
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

# custom libraries
import plotting_misc as plotmisc


##############################################################################
# %% Fake data

def create_fake_video_data(plot=False):
	""" Creates a fake hall thruster video for testing of the various functions """

	# initialize
	np.random.seed(0)
	x = np.arange(192)
	y = np.arange(152)
	dt = 5.714285714285713e-06
	t = np.arange(0.022857142857142857,
				0.028857142857142856 + dt / 2, 5.714285714285713e-06)
	x0 = 99.6
	y0 = 73.3

	# 3D grid
	T, Y, X = np.meshgrid(t, y, x, indexing='ij')
	R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
	Theta = np.arctan2(Y - y0, X - x0)

	def channel_time_average(X=X, Y=Y, T=T, a=1000, w=6.5, r0=57.5):
		return a * np.exp(-0.5 * ((R - r0) / w) ** 2)

	def channel_modes(m, X=X, Y=Y, T=T, a=1000, w=6.5, r0=57.5,
					f=10e3, phase=0):
		phi = -m * Theta - 2 * np.pi * f * T + phase
		return a * np.exp(-0.5 * ((R - r0) / w) ** 2) * np.sin(phi)

	def cathode_time_average(X=X, Y=Y, T=T, a=1000, w=2.5):
		return a * np.exp(-0.5 * (R / w) ** 2)

	def cathode_modes(m, X=X, Y=Y, T=T, a=419, w=4.5, f=10e3, 
					phase=0):
		phi = -m * Theta - 2 * np.pi * f * T + phase
		return a * np.exp(-0.5 * (R / w) ** 2) * np.sin(phi)

	def noise(sigma=10, shape=X.shape):
		v = np.random.normal(0, sigma, shape)
		v = np.sqrt(v ** 2)
		return v

	# create video
	x1 = noise(10) + channel_time_average() + cathode_time_average()
	x2 = channel_modes(2, f=2.4e3, a=25) + channel_modes(3, f=7.53e3, a=125)
	x3 = channel_modes(0, a=150, f=20e3) + cathode_modes(1, f=80e3, a=50)
	x4 = cathode_modes(0, f=38.2e3, a=25)
	video = xr.DataArray(x1 + x2 + x3 + x4,
						dims=['t', 'y', 'x'], coords=[t, y, x])
	video -= video.min()

	if plot:
		plt.figure()
		video[0, :, :].plot()

	return video


##############################################################################
# %% Video processing, misc

def calc_video_time_average(video, plot=False):
	""" 
	calculates the time average of frame of the video 
	
	Example
	-------
	
	Example 1::
		
		video = create_fake_video_data()
		calc_video_time_average(video, plot=True)
		
	"""
	
	ave = video.mean(dim='t')
	if plot:
		fig, ax = plt.subplots()
		ave.plot(ax=ax)
		ax.set_title('time average')
	return ave


def calc_channel_average_intensity(video, ri=0.975, ro=1.025):
	""" 
	Calculates the time-averaged channel intensity
	between inner radius (ri) and outer radius (ro)
	
	Example
	-------
	
	Example 1::
		
		video = create_fake_video_data()
		video_scaled, _ = scale_video_spatial(video, method='gaussian', plot=False)
		print(calc_channel_average_intensity(video_scaled))
	"""

	mask = create_radial_mask(video, ri=ri, ro=ro)
	video_time_average = calc_video_time_average(video)
	average_channel_intensity = np.nanmean(video_time_average * mask)

	return average_channel_intensity


def create_radial_mask(video, ri=0.9, ro=1.1, fillValue=np.nan, plot=False):
	"""
	Creates a radial mask between ri and ro

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
	plot : bool
		Optional plot

	Returns
	-------
	mask : numpy.ndarray (2D)
		Mask with 1s in the "keep" region and fillValue in the "masked-out" region

	Examples
	--------
	
	Example 1::
		
		video = create_fake_video_data()
		video_scaled, _ = scale_video_spatial(video, method='gaussian', plot=False)
		create_radial_mask(video_scaled, plot=True)
	"""

	R, _ = calc_video_polar_coordinates(video)
	mask = np.ones(R.shape)
	mask[(R > ro) | (R < ri)] = fillValue

	if plot:
		temp = xr.DataArray(mask, dims=['y', 'x'],
							coords=[video.y, video.x])
		fig, ax = plt.subplots()
		temp.plot(ax=ax)

	return mask


def calc_video_polar_coordinates(video, plot=False):
	"""
	Creates polar coordinates for the video from the video's cartesian 
	coordinates
	
	Examples
	--------
	
	Example 1::
		
		video = create_fake_video_data()
		video_scaled, _ = scale_video_spatial(video, method='gaussian', plot=False)
		calc_video_polar_coordinates(video_scaled, plot=True)
	"""

	# Convert 1D x and y arrays to 2D matrices
	X, Y = np.meshgrid(video.x, video.y)
	
	# calculate R and Theta from X and Y
	R = np.sqrt(X ** 2 + Y ** 2)
	Theta = np.arctan2(Y, X)

	if plot:
		X = xr.DataArray(X, dims=['y', 'x'], coords=[video.y, video.x])
		Y = xr.DataArray(Y, dims=['y', 'x'], coords=[video.y, video.x])
		R_temp = xr.DataArray(R, dims=['y', 'x'], coords=[video.y, video.x])
		Theta_temp = xr.DataArray(Theta, dims=['y', 'x'],
								coords=[video.y, video.x])
		fig, ax = plt.subplots(1, 4)
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


##############################################################################
# %% Azimuthal binning


def solve_for_bin_edges(numberBins=100):
	""" creates bin edges for the specified number of bins """
	return np.linspace(-np.pi, np.pi, numberBins + 1)


def azimuthal_binning(video, numberBins, ri, ro, plot=False):
	"""
	Azimuthally bins the video values between ri and ro
	
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
	
	Example 1::
		
		video = create_fake_video_data()
		video_scaled, _ = scale_video_spatial(video, method='gaussian', plot=False)
		azimuthal_binning(video_scaled, 100, 0.9, 1.1, plot=True)
	"""

	def bin_data_and_average(x, y, numberBins, plot=False):
		"""
		Bins data and averages the values within each bin

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

			x = np.linspace(-np.pi, np.pi, 1000)
			y = np.cos(x) + 1 * (np.random.rand(x.shape[0]) - 0.5)
			numberBins = 100
			bin_results = bin_data_and_average(x, y, numberBins, plot=True)

		"""
		bin_edges = solve_for_bin_edges(numberBins)

		# bin y(x) into discrete bins and average the values within each
		y_binned, _, _ = binned_statistic(x, y, bins=bin_edges,
										statistic='mean')
		x_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

		if plot:
			da_raw = xr.DataArray(y, dims=['x'], coords=[x]).sortby('x')
			fig, ax = plt.subplots()
			da_raw.plot(ax=ax, label='raw data')
			ax.plot(x_bins, y_binned, label='binned data',
					marker='s', ms=3, linestyle='--')
			ax.legend()

		return xr.DataArray(y_binned, dims='Theta', coords=[x_bins])

	# create radial mask
	R, Theta = calc_video_polar_coordinates(video)
	mask = create_radial_mask(video, ri=ri, ro=ro)

	# bin and average each time step in the data
	binned_data = np.zeros((video.t.shape[0], numberBins))
	for i, t in enumerate(video.t.data):
		unbinned_data = pd.DataFrame()
		unbinned_data['theta'] = Theta.reshape(-1)
		unbinned_data['radius'] = R.reshape(-1)
		unbinned_data['data'] = (video.sel(t=t).data * mask).reshape(-1)
		unbinned_data = unbinned_data.dropna()

		if i == 0 and plot:
			plot2 = True
		else:
			plot2 = False

		out = bin_data_and_average(unbinned_data.theta.values,
									unbinned_data.data.values,
									numberBins, plot=plot2)
		binned_data[i, :] = out

	binned_data = xr.DataArray(binned_data, dims=['t', 'theta'],
							coords=[video.t.data.copy(), out.Theta])

	if plot:
		fig, ax = plt.subplots()
		binned_data.plot(ax=ax)

	return binned_data


##############################################################################
# %% Scale video


def scale_video_amplitude(video, method='std'):
	""" 
	scales (centers and normalizes) the video's amplitude
	by the channel average or each pixel's std
	
	Examples
	--------
	
	Example 1::
		
		video = create_fake_video_data()
		video_scaled, _ = scale_video_spatial(video, method='gaussian', plot=False)
		video_scaled2 = scale_video_amplitude(video_scaled, method='std')
		fig, ax = plt.subplots()
		video_scaled2[0, :, :].plot(ax=ax)
		
		video_scaled2 = scale_video_amplitude(video_scaled, method='channel average')
		fig, ax = plt.subplots()
		video_scaled2[0, :, :].plot(ax=ax)
	"""
	if method == 'std':
		out = (video - calc_video_time_average(video)) / video.std(dim='t')
		return out.fillna(0)

	elif 'channel' in method or 'average' in method:
		I_avg = calc_channel_average_intensity(video)
		return (video - calc_video_time_average(video)) / I_avg

	else:
		raise Exception("""Improper method provided.  Encountered: %s"""
						% str(method))


def scale_video_spatial_hough(images, rmin=35, rmax=70, sigma=3,
							low_threshold=10, high_threshold=50, plot=False):
	"""
	Scale (center and normalize) the video's cartesian
	coordinates using the hough-circle method

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	rmin : int, optional
		Smallest acceptable radius.  Units in pixel count
	rmax : int, optional
		Largest acceptable radius.  Units in pixel count
	sigma : float, optional
		Edge detection parameter. The default is 3.
	low_threshold : float, optional
		Edge detection parameter.  The default is 10.
	high_threshold : float, optional
		Edge detection parameter. The default is 50.
	plot : bool
		optional plot of the results

	Returns
	-------
	video : xarray.core.dataarray.DataArray
		the video with coordinates scaled
	fit_params : dict
		Fit parameters

	References
	----------
	* https://scikit-image.org/docs/dev/auto_examples/edges/
		plot_circular_elliptical_hough_transform.html

	Examples
	--------
	Example 1 ::

		video = create_fake_video_data()
		video_scaled, params = scale_video_spatial_hough(video, plot=True)
	"""

	# convert video to time averaged image
	image = calc_video_time_average(images).data

	# edge detection
	edges = canny(image, sigma=sigma,
				low_threshold=low_threshold, high_threshold=high_threshold)

	# Detect circles between min and max radii
	hough_radii = np.arange(rmin, rmax, 1)
	hough_res = hough_circle(edges, hough_radii)

	# Select the two most prominent circles (inner and outer edges of channel?)
	accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
											total_num_peaks=2)

	# Optional plot of results
	if plot:
		fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
		ax[0].imshow(image)
		ax[1].imshow(edges)
		for center_y, center_x, radius in zip(cy, cx, radii):
			plotmisc.circle(ax[0], xy=(center_x, center_y), r=radius, fill=False)
			plotmisc.circle(ax[1], xy=(center_x, center_y), r=radius, fill=False)
		ax[0].set_title('average image with fit')
		ax[1].set_title('image edges with fit')

	# results
	fit_params = {'a0': 0, 'x0': cx.mean(), 'y0': cy.mean(),
				'r0': radii.mean(),
				'sigma0': (radii.max() - radii.min()) / 2, 'offset': 0}

	# apply correction to the video
	images = images.copy()
	images['x'] = (images.x - fit_params['x0']) / fit_params['r0']
	images['y'] = (images.y - fit_params['y0']) / fit_params['r0']

	return images, fit_params


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
	X, Y = np.meshgrid(image.x.data, image.y.data)

	# annular Gaussian model, assumed form of the channel
	def model(image, params):
		a0, x0, y0, r0, sigma0, offset = params

		def gaussian(a, r, sigma, R):
			return a * np.exp(-0.5 * ((R - r) / sigma) ** 2)

		R0 = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
		Z = gaussian(a0, r0, sigma0, R0) ** 1 + offset

		return Z

	# Generate a reasonable guess and guess image
	if len(guess) < 6:
		sh = image.shape
		guess = [1, sh[1] // 2, sh[0] // 2, np.min(sh) / 3, np.min(sh) / 4, 4]

	# Function that minimizes (i.e. fits) the parameters to the model
	def min_func(params):
		Z = model(image.data, params)
		error = np.abs((image.data - Z)).sum()
		if verbose:
			print('error = %.6f' % error)
		return error

	# perform fit
	fit = minimize(min_func, guess)
	a0, x0, y0, r0, sigma0, offset = fit.x
	fit_params = {'a0': a0, 'x0': x0, 'y0': y0, 'r0': r0,
				'sigma0': sigma0, 'offset': offset}

	# optional plot of results
	if plot:
		Z_fit = xr.DataArray(model(image, fit.x),
							dims=image.dims, coords=image.coords)
		Z_guess = xr.DataArray(model(image, guess),
							dims=image.dims, coords=image.coords)

		fig, ax = plt.subplots(1, 2, sharey=True)
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

		fig0, ax0 = plt.subplots(1, 4)

		ax0[0].imshow(image, origin='lower')
		ax0[0].set_title('actual')

		ax0[1].imshow(Z_guess, origin='lower')
		ax0[1].set_title('guess')

		ax0[2].imshow(Z_fit, origin='lower')
		ax0[2].set_title('fit')

		ax0[3].imshow(image, origin='lower')
		ax0[3].set_title('actual with fit')

		plotmisc.circle(ax0[3], xy=(x0, y0), r=r0, fill=False, linestyle='--')
		plotmisc.circle(ax0[3], xy=(x0, y0), r=r0 + sigma0 * 1.5, fill=False)
		plotmisc.circle(ax0[3], xy=(x0, y0), r=r0 - sigma0 * 1.5, fill=False)

	# apply correction to the video
	video = video.copy()
	video['x'] = (video.x - x0) / r0
	video['y'] = (video.y - y0) / r0

	return video, fit_params


def scale_video_spatial(video, method='gaussian', plot=False):
	""" scales (centers and normalizes) the videos cartesian coordinates
		using an annular gaussian method or a hough-circle method
	"""

	if method == 'gaussian' or 'aussian' in method:
		return scale_video_spatial_gaussian(video, plot=plot)
	elif method == 'hough' or 'ough' in method:
		return scale_video_spatial_hough(video, plot=plot)
	else:
		raise Exception("""Improper method provided.
						Instead encountered: %s""" % method)
						
											
##############################################################################
# %% POD mode analysis

						
def calc_POD_bases(video, plot=False, saveAllFigs=False, num_bases=1000, figsName='SVD_analysis.pdf'):
	"""
	Apply the POD-SVD video decomposition algorithm to the video
	and then return the properly formatted topos, energy, and chronos

	Parameters
	----------
	video : xarray.core.dataarray.DataArray
		the video
	plot : bool
		optional plot of results
	saveAllFigs : bool
		optionally saves all plots as a PDF

	Returns
	-------
	topos : xarray.core.dataarray.DataArray
		POD topos
	Sigma : xarray.core.dataarray.DataArray
		POD energy
	chronos : xarray.core.dataarray.DataArray
		POD chronos

	Examples
	--------

	Example 1 ::

		video = create_fake_video_data()
		calc_POD_bases(video, plot=True)

	"""

	# SVD subfunction
	def svdVideoDecomposition(X, plot=False, saveFigs=False,
							figsName='SVD_analysis.pdf', fileRoot='',
							tlim=None, nperseg=5, numBasesToPlot=30):
		"""

		Parameters
		----------
		X : xarray.core.dataarray.DataArray
			Video data in 3D xarray DataArray format
			with dimensions ('t','y','x').

		References
		----------
		*  Algorithm is spelled out in Tu2014
		https://www.aimsciences.org/article/doi/10.3934/jcd.2014.1.391

		"""

		# Make sure input is formatted correctly
		if np.isnan(X.data).any():
			raise Exception('Input signal contains NaN')
		if 'time' in X.dims:
			X = X.rename({'time': 't'})
		if 'x' not in X.dims and 'y' not in X.dims and 't' not in X.dims:
			raise Exception("""Dimensions not formatted correctly.
							Should be (t,x,y)""")

		# reshape 3D data to 2D in the desired order = (space, time)
		X = X.stack(z=('y', 'x')).transpose('z', 't')

		# svd algorithm
		try:
			U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
		except np.linalg.LinAlgError:
			U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

		# clip results based on minimum dimension
		n_rank = min(min(X.shape), num_bases)
		U = U[:, :n_rank]
		Sigma = Sigma[:n_rank]
		VT = VT[:n_rank, :]

		# save results as xarray dataarrays
		energy = xr.DataArray(Sigma, dims=['basisNum'],
							coords={'basisNum': np.arange(n_rank)})

		topos = xr.DataArray(U, dims=['z', 'basisNum'],
							coords={'z': X.z, 'basisNum':
									np.arange(n_rank)}).unstack('z')

		chronos = xr.DataArray(VT, dims=['basisNum', 't'],
							coords={'t': X.t,
									'basisNum': np.arange(n_rank)})

		# optional plots
		if plot:
			plt.ioff()
			figs = []

			# Energy plot
			fig, ax = plt.subplots(2, sharex=True)
			energy.plot(ax=ax[0], marker='.')
			ax[0].set_yscale('log')
			ax[0].set_title('Basis energy')
			plotmisc.subTitle(ax[0], 'Basis energy')
			ax[0].set_ylabel('Energy (a.u.)')
			a = np.cumsum(energy) / energy.sum()
			ax[1].plot(energy.basisNum, a, marker='.')
			ax[1].set_ylabel('Cumulative sum (normalized)')
			ax[1].set_xlabel('Basis number')
			plotmisc.subTitle(ax[1], 'Cumulative sum of mode energy (normalized)')
			figs.append(fig)

			# Bases plots
			for i in range(numBasesToPlot):
				fig, ax = plt.subplots(1, 3)
				N = chronos.t.shape[0] // int(nperseg)
				da_fft = fft_average(chronos[i, :] * np.sqrt(Sigma[i]),
									nperseg=N,
									noverlap=np.floor(N * 0.9).astype(int),
									plot=False, trimNegFreqs=True,
									zeroTheZeroFrequency=True, returnAbs=True,
									verbose=False)
				da_fft['f'] = da_fft.f * 1e-3
				f_max = fft_max_freq(da_fft)
				(topos.sel(basisNum=i) * np.sqrt(energy.sel(
												basisNum=i))).plot(ax=ax[0])
				ax[0].set_aspect('equal')
				(chronos.sel(basisNum=i) * np.sqrt(energy.sel(
												basisNum=i))).plot(ax=ax[1])
				if isinstance(tlim, type(None)):
					ax[1].set_xlim(tlim)
				da_fft.plot(ax=ax[2])
				ax[2].set_title(ax[1].get_title() + ', peak freq. = %.2f kHz' %
								f_max)
				ax[2].set_xlabel('kHz')
				ax[2].set_ylabel('Spectral power (au)')
				plotmisc.finalizeFigure(fig, figSize=[12, 4])
				plotmisc.subTitle(ax[0], 'spatial mode', xy=(0.98, 0.98), horizontalalignment='right')
				plotmisc.subTitle(ax[1], 'mode evolution', xy=(0.98, 0.98), horizontalalignment='right')
				plotmisc.subTitle(ax[2], 'FFT(mode evolution)', xy=(0.98, 0.98), horizontalalignment='right')
				figs.append(fig)
				if saveFigs:
					plt.close(fig)

			if saveFigs:
				plotmisc.save_list_of_figures_to_pdf(figs, figsName)

			plt.ion()
			plt.show()

		return topos, chronos, energy

	# main
	video = video.copy()

	topos, chronos, Sigma = svdVideoDecomposition(video, plot=plot,
												saveFigs=saveAllFigs,
												numBasesToPlot=20,
												figsName=figsName)
	topos['x'] = video.x.values
	topos['y'] = video.y.values
	
	return topos, Sigma, chronos

def plot_POD_results(topos, Sigma, chronos, basis_num=0, nperseg=5):
	i = basis_num
	energy = Sigma
	
	fig, ax = plt.subplots(1, 3)
	N = chronos.t.shape[0] // int(nperseg)
	da_fft = fft_average(chronos[i, :] * np.sqrt(Sigma[i]),
						nperseg=N,
						noverlap=np.floor(N * 0.9).astype(int),
						plot=False, trimNegFreqs=True,
						zeroTheZeroFrequency=True, returnAbs=True,
						verbose=False)
	da_fft['f'] = da_fft.f * 1e-3
	f_max = fft_max_freq(da_fft)
	(topos.sel(basisNum=i) * np.sqrt(energy.sel(
									basisNum=i))).plot(ax=ax[0])
	ax[0].set_aspect('equal')
	(chronos.sel(basisNum=i) * np.sqrt(energy.sel(
									basisNum=i))).plot(ax=ax[1])
# 	if isinstance(tlim, type(None)):
# 		ax[1].set_xlim(tlim)
	da_fft.plot(ax=ax[2])
	ax[2].set_title(ax[1].get_title() + ', peak freq. = %.2f kHz' %
					f_max)
	ax[2].set_xlabel('kHz')
	ax[2].set_ylabel('Spectral power (au)')
	plotmisc.finalizeFigure(fig, figSize=[12, 4])
	plotmisc.subTitle(ax[0], 'spatial mode')
	plotmisc.subTitle(ax[1], 'mode evolution')
	plotmisc.subTitle(ax[2], 'FFT(mode evolution)')

##############################################################################
# %% Signal processing, misc


def signal_spectral_properties(da, nperseg=None, verbose=True):

	# check input
	if 'time' in da.dims:
		da = da.rename({'time': 't'})
	if 't' not in da.dims:
		raise Exception('Time dimension, t, not present.  Instead, %s found'
				% (str(da.dims)))
	
	# preliminary steps
	params = {}
	params['dt'] = float(da.t[1] - da.t[0])
	params['f_s'] = 1.0 / params['dt']
	if type(nperseg) != int:
		nperseg = da.t.shape[0]
	if verbose: 
		print('Window size: %d' % nperseg)
	
	# Nyquist frequency (highest frequency)
	f_nyquist = params['f_s'] / 2.
	params['f_nyquist'] = f_nyquist
	if verbose: 
		print("Nyquist freq., %.2f" % f_nyquist)
	
	# time window
	time_window = params['dt'] * nperseg
	params['time_window'] = time_window
	if verbose: 
		print("Time window: %.3e s" % time_window)
	
	# frequency resolution, also the lowest frequency to get a full wavelength
	params['f_res'] = params['f_s'] / nperseg
	if verbose: 
		print("Frequency resolution, %.2f" % params['f_res'])

	return params # dt, f_s, f_nyquist, time_window, f_res


# def coherenceAnalysis(	da1,
# 						da2,
# 						numPointsPerSegment=1024,
# 						plot=False,
# 						noverlap=None,
# 						verbose=True,
# 						removeOffsetAndNormalize=False):
# 	"""
# 	Coherence analysis between two signals. Wrapper for scipy's coherence analysis
# 	
# 	Parameters
# 	----------
# 	da1 : xarray.DataArray
# 		time dependent data signal 1
# 	da2 : xarray.DataArray
# 		time dependent data signal 1
# 	numPointsPerSegment : int
# 		default 1024. number of points used in the moving analysis
# 	plot : bool
# 		True = plots the results
# 	noverlap : int
# 		default None. number of points-overlapped in the moving analysis
# 	verbose : bool
# 		If True, plots misc parameters.
# 		
# 	Returns
# 	-------
# 	coh : xarray.DataArray
# 		coherence analysis
# 	
# 	References
# 	----------
# 	* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
# 	
# 	Example 1
# 	---------
# 	::
# 		
# 		# Case where signals have the same phase (with a different offset) and one signal has a lot of noise
# 		import numpy as np
# 		import matplotlib.pyplot as plt; plt.close('all')
# 		freq=5
# 		time = np.arange(0, 10, 0.01)
# 		phase_signal_1=time*np.pi*2*freq
# 		phase_signal_2=time*np.pi*2*freq+0.5
# 		np.random.seed(0)
# 		da1 = _xr.DataArray(	np.sin(phase_signal_1)+ 1.5*np.random.randn(len(time)),
# 									dims=['t'],
# 									coords={'t':time},
# 									name='signal1'	)*2+1
# 		da1.attrs=	{'long_name':'signal1',
# 						'units':'au'}
# 		da2 = _xr.DataArray(	np.sin(phase_signal_2)+ 0.0*np.random.randn(len(time)),
# 									dims=['t'],
# 									coords={'t':time},
# 									name='signal2'	)
# 		da2.attrs=	{'long_name':'signal2',
# 						'units':'au'}
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=0)
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=64)
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=127)


# 	Example 2
# 	---------
# 	::
# 		
# 		# Case where signals have the same phase (with a different offset), one signal has two frequency components and noise
# 		import numpy as np
# 		import matplotlib.pyplot as plt; plt.close('all')
# 		freq=5
# 		time = np.arange(0, 10, 0.01)
# 		phase_signal_1=time*np.pi*2*freq
# 		phase_signal_2=time*np.pi*2*freq+0.5
# 		np.random.seed(0)
# 		da1 = _xr.DataArray(	np.sin(phase_signal_1),
# 									dims=['t'],
# 									coords={'t':time},
# 									name='signal1'	)
# 		da2 = _xr.DataArray(	np.sin(phase_signal_2) + 2*np.sin(time*np.pi*2*17.123) + 1.20*np.random.randn(len(time)),
# 									dims=['t'],
# 									coords={'t':time},
# 									name='signal2'	)
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=0)
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=64)
# 		coh=coherenceAnalysis(da1,da2,plot=True,numPointsPerSegment=128,noverlap=127)

# 	"""
# 	
# 	def _coherenceComplex(da1, da2, window='hann', nperseg=None, noverlap=None,
# 				nfft=None):
# 		r"""
# 		This subfunction is a modification of the scipy.linspace.coherence 
# 		function with a few modifications. These include using xarrays as 
# 		input and output and returning the complex coherence instead of the 
# 		absolute value.  
# 		"""
# 		import scipy.signal as signal
# 		import numpy as np
# 		
# 		detrend = 'constant'
# 		axis = -1
# 		
# 		dt = float(da1.t[1] - da1.t[0])
# 		fs = 1.0 / dt
# 		freqs, Pxx = signal.welch(da1, fs=fs, window=window, nperseg=nperseg,
# 						noverlap=noverlap, nfft=nfft, detrend=detrend,
# 						axis=axis)
# 		_, Pyy = signal.welch(da2, fs=fs, window=window, nperseg=nperseg, 
# 							noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
# 		_, Pxy = signal.csd(da1, da2, fs=fs, window=window, nperseg=nperseg,
# 					noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
# 	
# 		Cxy = Pxy / np.sqrt(Pxx * Pyy)
# 		Cxy *= Cxy
# 	
# 		return xr.DataArray(Cxy,
# 							dims=['f'],
# 							coords={'f': freqs})
# 	
# 	import matplotlib.pyplot as plt
# 	import numpy as np
# 	
# 	if 'time' in da1.dims:
# 		da1 = da1.rename({'time': 't'})
# 	if 'time' in da2.dims:
# 		da2 = da2.rename({'time': 't'})
# 	if 't' not in da1.dims and 't' not in da2.dims:
# 		raise Exception('Time dimension, t, not present. \
# 				Instead, %s found' % (str(da1.dims) + str(da2.dims)))
# 		
# 	if removeOffsetAndNormalize is True:
# 		da1 = (da1.copy() - da1.mean()) / da1.std()
# 		da2 = (da2.copy() - da2.mean()) / da2.std()
# 	
# 	dt, fs, _, _, f_min = signal_spectral_properties(da1, nperseg=numPointsPerSegment, verbose=verbose).values()
# 	
# 	# coherence analysis 
# 	coh = _coherenceComplex(da1, da2, nperseg=numPointsPerSegment,
# 						noverlap=noverlap, nfft=numPointsPerSegment * 2)
# 	
# 	# optional plot
# 	if plot is True:
# 		fig, (ax1A, ax2, ax3) = plt.subplots(nrows=3)
# 		ax1B = ax1A.twinx()
# 		da1.plot(ax=ax1A, label=da1.name)
# 		da2.plot(ax=ax1B, label=da2.name, color='tab:blue')
# 		ax1A.legend(loc='upper left')
# 		ax1B.legend(loc='upper right')
# 		ax1A.set_xlabel('time (s)')
# 		ax1A.set_ylabel(da1.name)
# 		ax1B.set_ylabel(da2.name)
# 		
# 		f1 = fft_average((da1 - da1.mean()) / da1.std(), 
# 				nperseg=numPointsPerSegment, verbose=False)
# 		np.abs(f1).where(f1.f >= f_min).plot(ax=ax2, label=da1.name, yscale='log')
# 		f2 = fft_average((da2 - da2.mean()) / da2.std(), 
# 				nperseg=numPointsPerSegment, verbose=False)
# 		np.abs(f2).where(f2.f >= f_min).plot(ax=ax2, label=da2.name, yscale='log')
# 		ax2.legend()
# 		
# 		np.abs(coh).plot(ax=ax3, label='Coherence', marker='', linestyle='-')
# 		ax3.set_xlabel('frequency (Hz)')
# 		ax3.set_ylabel('Coherence')
# 		ax3.legend()
# 		ax3.set_ylim([0, 1])
# 		
# 		if noverlap is None:
# 			noverlap = 0
# 		ax1A.set_title('points overlapped = \n %d of %d' % (noverlap, numPointsPerSegment))
# 		plotmisc.finalizeFigure(fig)
# 		
# 	return coh


def filtfilt(da, cornerFreq, filterType='low', filterOrder=1,
			plot=False, axis='t', time=None):
	"""
	Forward-backwards filter using a butterworth filter

	Parameters
	----------
	da : xarray.core.dataarray.DataArray
		Signal.  Index is time with units in seconds.
	cornerFreq : float or numpy.array of floats
		Filter's corner frequency
		If bandpass or bandstop, cornerFreq is a numpy.array with two elements
	filterType : str
		* 'low' - low-pass filter
		* 'high' - high-pass filter
		* 'bandpass'
		* 'bandstop'
	filterOrder : int
		Order of the butterworth filter.  8 is default.
	plot : bool
		Optional plot
	time : numpy array, optional
		if da is a numpy array, then time needs to be provided

	Returns
	-------
	dfOut : xarray.core.dataarray.DataArray
		Filtered signal.  Index is time with units in seconds.

	Notes
	-----
	* Be careful with high filterOrders.
	* I've seen them produce nonsense results.
	* I recommend starting low and then turning up the order

	Examples
	--------
	Example 1::

		import numpy as np
		import xarray as xr
		t = np.linspace(0, 1.0, 2001)
		dt = t[1] - t[0]
		fs = 1.0 / dt
		xlow = np.sin(2 * np.pi * 5 * t)
		xmid = np.sin(2 * np.pi * 50 * t)
		xhigh = np.sin(2 * np.pi * 500 * t)
		x = xlow + xhigh + xmid
		da = xr.DataArray(x, dims=['t'], coords={'t': t})
		dfOut = filtfilt(da, cornerFreq=30.0, filterType='low',
						filterOrder=8, plot=True)
		dfOut = filtfilt(da, cornerFreq=200.0, filterType='high',
						filterOrder=8, plot=True)
		dfOut = filtfilt(da, cornerFreq=np.array([25, 150]),
						filterType='bandpass', filterOrder=4, plot=True)
		dfOut = filtfilt(x, cornerFreq=np.array([25, 150]),
						filterType='bandpass', filterOrder=4,
						plot=True, time=t)

	References
	----------
	* https://docs.scipy.org/doc/scipy-0.14.0/
	*	reference/generated/scipy.signal.filtfilt.html

	"""
	if type(da) not in [xr.core.dataarray.DataArray]:
		if type(da) in [np.ndarray]:
			da = xr.DataArray(da, dims=['t'], coords={'t': time})
		else:
			raise Exception('Invalid input type')
	if 'time' in da.dims:
		da = da.rename({'time': 't'})
	if 't' not in da.dims:
		raise Exception('time not present')

	# construct butterworth filter
	samplingFreq = float(1.0 / (da.t[1] - da.t[0]))
	# I don't know why this factor of 2 needs to be here
	Wn = np.array(cornerFreq).astype(float) / samplingFreq * 2
	b, a = sig.butter(filterOrder, Wn=Wn, btype=filterType, analog=False)

	# perform forwards-backwards filter
	daOut = xr.DataArray(sig.filtfilt(b, a, da.values.reshape(-1),),
						dims='t', coords={'t': da.t})

	if plot:
		fig, ax = plt.subplots()
		da.plot(ax=ax, label='original')
		daOut.plot(ax=ax, label='filtered')
		plotmisc.finalizeSubplot(ax, xlabel='Time', ylabel='Signal amplitude',)

	return daOut


def hilbertTransform(da, plot=False):
	"""
	Hilbert transform.  Shifts signal by 90 deg. and finds the phase between
	the imaginary and real components

	Parameters
	----------
	da : xarray.DataArray
		input signal
	plot : bool
		optional plot

	Returns
	-------
	daHilbert : pandas.core.frame.DataFrame
		hilbert transformed signals
	daAmp : pandas.core.frame.DataFrame
		amplitude of hilbert transformed signals
	daPhase : pandas.core.frame.DataFrame
		phase of hilbert transformed signals

	Examples
	--------
	Example 1::

		import numpy as np

		dt = 2e-6
		t = np.arange(0, 10e-3, dt)
		f = 1.5e3
		from johnspythonlibrary2.Process.SigGen import gaussianNoise
		y = np.sin(2 * np.pi * t * f + 0.25 * np.pi) + 0.001 * gaussianNoise(t)

		da = xr.DataArray(y, dims=['t'], coords={'t': t})
		daHilbert, daAmp, daPhase = hilbertTransform(da, plot=True)
	"""
	if type(da) != xr.core.dataarray.DataArray:
		raise Exception('Input not formatted correctly')
	if 't' not in da.dims:
		raise Exception('Time coordinate not formatted correctly')

	daHilbert = xr.DataArray(sig.hilbert(da.copy() - da.mean()),
							dims=['t'], coords={'t': da.t})
	daAmp = np.abs(daHilbert)
	daPhase = xr.DataArray(np.arctan2(np.imag(daHilbert), np.real(daHilbert)))

	if plot:
		dt = (da.t[1] - da.t[0]).data
		daFreq = xr.DataArray(np.gradient(np.unwrap(daPhase),
										dt) / (2 * np.pi),
							dims=['t'], coords={'t': da.t})

		fig, ax = plt.subplots(3, sharex=True)
		np.real(daHilbert).plot(ax=ax[0], label='original')
		np.imag(daHilbert).plot(ax=ax[0], label=r'90$^o$ shifted')
		daAmp.plot(ax=ax[0], label=r'amplitude')
		daPhase.plot(ax=ax[1], label=r'phase', linestyle='', marker='.')
		daFreq.plot(ax=ax[2], label=r'freq')

		plotmisc.finalizeSubplot(ax[0], ylabel='Amplitude',
						subtitle='Signals', legendLoc='lower right')
		plotmisc.finalizeSubplot(ax[1], ylabel='Rad.', subtitle='Phase',
							legendOn=False,
							yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
							ytickLabels=[r'-$\pi$', r'-$\pi$/2',
										r'0', r'$\pi$/2', r'$\pi$'],
							ylim=[-np.pi, np.pi], xlabel='Time',)
		plotmisc.finalizeSubplot(ax[2], ylabel='Hz', subtitle='Frequency',
							xlabel='Time', legendOn=False)

	return daHilbert, daAmp, daPhase


def fft_max_freq(da_fft, positiveOnly=True):
	"""
	Calculates the maximum frequency associted with the fft results

	Parameters
	----------
	da_fft : pandas dataframe or xarray dataarray
	
	Example
	-------
	Example 1::

		import numpy as np

		dt = 2e-6
		t = np.arange(0, 10e-3, dt)
		f = 21e3
		y = np.sin(2 * np.pi * t * f + 0.25 * np.pi) + 2 * np.random.normal(size=len(t))
		da = xr.DataArray(y, dims=['t'], coords={'t': t})
		
		fft_results = fft_average(da, plot=False, nperseg=500, noverlap=400)
		print(fft_max_freq(fft_results))

	"""
	da_fft = np.abs(da_fft.copy())

	# if input is a xarray
	if type(da_fft) == xr.core.dataarray.DataArray:
		if positiveOnly:
			da_fft = da_fft[da_fft.f >= 0]

		return float(da_fft.idxmax().data)

	# if input is pandas
	elif (isinstance(da_fft, pd.core.frame.DataFrame) or isinstance(da_fft, pd.core.frame.Series)):
		if positiveOnly:
			da_fft = da_fft[da_fft.index > 0]

		return da_fft.idxmax(axis=0)

	# else raise an exception
	else:
		raise Exception('Invalid input type.  Encountered : %s' %
						str(type(da_fft)))


def fft_average(da, nperseg=None, noverlap=None, plot=False, verbose=True,
				trimNegFreqs=False, normalizeAmplitude=False,
				sortFreqIndex=False, returnAbs=False,
				zeroTheZeroFrequency=False, realAmplitudeUnits=False):
	"""
	Computes an averaged abs(fft) using Welch's method.
	This is mostly a wrapper for scipy.signal.welch

	Parameters
	----------
	da : xarray.DataArray or xarray.Dataset
		dataarray of time dependent data
		coord1 = time or t (units in seconds)
	nperseg : int, optional
		Length of each segment. Defaults to None, but if window is str
		or tuple, is set to 256, and if window is array_like,
		is set to the length of the window.
	noverlap : int, optional
		Number of points to overlap between segments.
		If None, noverlap = nperseg // 2. Defaults to None.
	plot : bool
		(optional) Plot results
	trimNegFreqs : bool
		(optional) True - only returns positive frequencies

	Returns
	-------
	da_fft : xarray.DataArray
		averaged abs(FFT) of da
		
	Example
	-------
	Example 1::

		import numpy as np

		dt = 2e-6
		t = np.arange(0, 10e-3, dt)
		f = 21e3
		y = np.sin(2 * np.pi * t * f + 0.25 * np.pi) + 2 * np.random.normal(size=len(t))
		da=xr.DataArray(y, dims=['t'], coords={'t': t})
		
		fft_average(da, plot=True, nperseg=500, noverlap=400)

	References
	-----------
	* https://docs.scipy.org/doc/scipy/reference/
		generated/scipy.signal.welch.html
	"""
	
	# check input
	if type(da) not in [xr.core.dataarray.DataArray, xr.core.dataset.Dataset]:
		raise Exception('Input data not formatted correctly')
	if type(da) in [xr.core.dataarray.Dataset]:
		return da.apply(fft_average, nperseg=nperseg, noverlap=noverlap,
						plot=plot, trimNegFreqs=trimNegFreqs,
						normalizeAmplitude=normalizeAmplitude,
						sortFreqIndex=sortFreqIndex,
						realAmplitudeUnits=realAmplitudeUnits,
						zeroTheZeroFrequency=zeroTheZeroFrequency,
						verbose=verbose)

	try:
		time = np.array(da.t)
	except AttributeError:
		raise Exception('Time dimension needs to be labeled t')

	# sampling rate
	dt = time[1] - time[0]
	fs = 1. / dt
	timeWindow = dt * nperseg
	if verbose:
		print("Time window: %.3e s" % timeWindow)

	# lowest frequency to get a full wavelength
	fLow = 1. / (nperseg * dt)
	if verbose:
		print("Lowest freq. to get 1 wavelength, %.2f" % fLow)

	# highest frequency
	nyF = fs / 2.
	if verbose:
		print("Nyquist freq., %.2f" % nyF)

	# perform fft-average
	dt = time[1] - time[0]
	freq, fft_abs = sig.welch(da.data, fs=(1.0 / dt), nperseg=nperseg,
							noverlap=noverlap,
							return_onesided=not(trimNegFreqs),
							scaling='spectrum')
	fft_results = xr.DataArray(fft_abs, dims=['f'], coords={'f': freq})
	fft_results.attrs["units"] = "au"
	fft_results.f.attrs["units"] = "Hz"
	fft_results.f.attrs["long_name"] = 'Frequency'
	fft_results.attrs["long_name"] = 'FFT amplitude'

	# options
	if realAmplitudeUnits:
		N = da.t.shape[0]
		fft_results *= 1.0 / N  # 2 / N if negative freqs have been trimmed
	if trimNegFreqs:
		fft_results = fft_results.where(fft_results.f >= 0).dropna(dim='f')
	if sortFreqIndex:
		fft_results = fft_results.sortby('f')
	if returnAbs:
		fft_results = np.abs(fft_results)
	if zeroTheZeroFrequency:
		fft_results.loc[0] = 0
	elif normalizeAmplitude:
		fft_results /= fft_results.sum()

	# optional plot of results
	if plot:
		da_temp = np.abs(fft_results.copy()).sortby('f')
		f, (ax1, ax2) = plt.subplots(nrows=2)
		da.plot(ax=ax1)
		da_temp.plot(ax=ax2)
		ax1.set_ylabel('Orig. signal')
		ax1.set_xlabel('Time')
		ax2.set_yscale('log')
		ax1.set_title(da_temp.name)
		ax2.set_ylabel('FFT Amplitude')
		ax2.set_xlabel('Frequency')

	return fft_results


##############################################################################
# %% Fourier mode analysis

def calc_1D_FFT_theta(images_az_binned, plot=False):
	"""
	Calculates the 1D FFT along the theta coordinate

	Examples
	--------
	
	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		video = scale_video_amplitude(video, method='std')
		images_az_binned = azimuthal_binning(video, numberBins=100,
											ri=0.9, ro=1.1)
		calc_1D_FFT_theta(images_az_binned, plot=True)
		
	"""
	images_az_binned = images_az_binned.copy()

	# check input
	if len(images_az_binned.dims) != 2:
		raise Exception("""Data needs to have only 2 dimensions.
						Instead, %d were encountered."""
						% len(images_az_binned.dims))
	if 't' not in images_az_binned.dims:
		raise Exception("""Data dimensions should be t and theta.
						Instead, %s were encountered."""
						% str(images_az_binned.dims))
	elif 'theta' not in images_az_binned.dims:
		raise Exception("""Data dimensions should be t and theta.
						Instead, %s were encountered."""
						% str(images_az_binned.dims))

	# preliminary steps
	dtheta = float(images_az_binned.theta[1] - images_az_binned.theta[0]) / (2 * np.pi)
	images_az_binned['t'] = images_az_binned.t * 1e3
	m = fftfreq(images_az_binned.shape[1], d=dtheta)

	# perform FFT
	X_1D = np.fft.fft(images_az_binned, axis=1)
	X_1D = xr.DataArray(X_1D, dims=['t', 'm'],
						coords=[images_az_binned.t, m]).sortby('m')

	# return the results to the correct amplitude
	N = images_az_binned.theta.shape[0]
	X_1D *= 2.0 / N  
	X_1D[:, N // 2] = X_1D[:, N // 2] / 2

	if plot:
		for m in range(5):  # first 5 bases only
			fig, ax = plt.subplots()
			X_1D.sel(m=m, method='nearest').real.plot(ax=ax,
													label='m=%d, real' % m)
			X_1D.sel(m=m, method='nearest').imag.plot(ax=ax,
													label='m=%d, imag' % m)
			ax.legend()

	return X_1D


def calc_2D_FFT_v2(images_az_binned, plot=False):
	"""
	Calculates the 2D FFT.  First standard FFT along the theta and then FFT-Welch along time.
	Optionaly plots the dispersion plot.

	Examples
	--------
	
	Example 1 ::

		video = create_fake_video_data()
		video, _ = scale_video_spatial_gaussian(video)
		video = scale_video_amplitude(video, method='std')
		images_az_binned = azimuthal_binning(video, numberBins=100,
											ri=0.9, ro=1.1)
		calc_2D_FFT_v2(images_az_binned, plot=True)
		
	"""

	X = calc_1D_FFT_theta(images_az_binned)

	# preliminary steps
	dt = float(X.t[1] - X.t[0])
	nperseg = 400
	freq, fft_abs = sig.welch(X.data, fs=1.0 / dt, nperseg=nperseg,
							noverlap=nperseg // 2, return_onesided=True,
							scaling='spectrum', axis=0)

	X_2D = xr.DataArray(fft_abs, dims=['f', 'm'],
						coords=[freq, X.m]).sortby('f')

	if plot:
		fig, ax = plt.subplots()
		np.log10(np.abs(X_2D)).plot(ax=ax)
		ax.set_title('dispersion plot')

	return X_2D
