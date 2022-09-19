"""
Functions related to the reading of photron highspeed videos
"""

import re as _re
import numpy as _np
import xarray as _xr


def read_12bit_photron_mraw_and_cihx_files(cihx_file_path, mraw_file_path=None, plot=False):
	"""
	Reads in 12 bit photron (monochrome) mraw and cihx video files and returns the video in an xarray.DataArray format.

	Parameters
	----------
	cihx_file_path : str
		Path of the cihx file.  Note that the mraw file should have the same name (just different extension)
	plot : bool, optional
		Plots the first few frames of the video

	Returns
	-------
	video : xarray.DataArray
		16 bit video with dimensions labeled.  

	"""
			
	def process_cihx_file(cihx_file_path, keys=['width', 'height', 'bit', 'totalFrame', 'startFrame', 'recordRate']):
		""" Extract various metadata values from the cihx file. """
		def extract_value(data, key):
			return _re.search('<%s>(.+?)</%s>' % (key, key), data).group(1)
		
		# get file contents
		f = open(cihx_file_path, "r", encoding="utf8", errors='ignore')
		data = f.read()
		f.close()
		data = '::'.join(data.split('\n')[1:])
		
		# search contents for each key and return values
		values = {}
		for key in keys:
			values[key] = extract_value(data, key)
			
		return values
	
	def read_uint12(data_chunk):
		""" 
		Convert 12 bit video data to 16 bit.
		
		References
		----------
		 * https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files
		"""
		data = _np.frombuffer(data_chunk, dtype=_np.uint8)
		fst_uint8, mid_uint8, lst_uint8 = _np.reshape(data, (data.shape[0] // 3, 3)).astype(_np.uint16).T
		fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
		snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
		return _np.reshape(_np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
		
	# extract metadata from cihx file
	cihx_values = process_cihx_file(cihx_file_path)
	
	# if not specified, assume the mraw file has the same name as the .cihx file
	if mraw_file_path is None:
		mraw_file_path = cihx_file_path.split('.')[0] + '.mraw'
	
	# open and process mraw file
	with open(mraw_file_path, mode='rb') as file: # b is important -> binary
	    fileContent = file.read()
	raw_data = read_uint12(fileContent)
	
	# process video into an xarray.DataArray
	x = _np.arange(int(cihx_values['width']))
	y = _np.arange(int(cihx_values['height']), 0, -1) - 1
	dt = 1 / float(cihx_values['recordRate'])				# time step
	t = (_np.arange(int(cihx_values['totalFrame'])) + int(cihx_values['startFrame'])) * dt
	x = _xr.DataArray(x, dims='x', coords=[x], attrs={'long_name': 'x', 'units':'pixel count'})
	y = _xr.DataArray(y, dims='y', coords=[y], attrs={'long_name': 'y', 'units':'pixel count'})
	t = _xr.DataArray(t, dims='t', coords=[t], attrs={'long_name': 't', 'units':'s'})
	video = _xr.DataArray(raw_data.reshape((len(t), len(y), len(x))), coords={'t': t, 'y': y, 'x': x}, attrs={'long_name': 'pixel intensity', 'units': 'au'})
	
	# (optional) plot several frames of the video
	if plot is True:
		import matplotlib.pyplot as plt
		number_of_frames = 3
		fig, axes = plt.subplots(number_of_frames)
		for i, ax in enumerate(axes):
			video[i, :, :].plot(ax=ax)
			ax.set_aspect('equal')
			
	return video


