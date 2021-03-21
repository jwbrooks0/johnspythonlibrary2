import pandas as _pd
import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr


#%% filter video

def filter_video(	da,
					cornerFreq,
					filterType='low',
					filterOrder=1,
					plot=False):
	
	## import
	from johnspythonlibrary2.Process.Filters import filtfiltWithButterworth
	
	## check input 
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 'x' not in da.dims and 'y' not in da.dims and 't' not in da.dims:
		raise Exception('Dimensions not formatted ocrrectly.  Should be (t,x,y)')
	da=da.transpose('t',...) # 't' dimension must be first.
	
	## filter
	out=_np.apply_along_axis(	filtfiltWithButterworth, 
								axis=0, 
								arr=da.data, 
								cornerFreq=cornerFreq,
								filterOrder=filterOrder,
								time=da.t.data,
								filterType=filterType)
	daFiltered=_xr.DataArray(	out,
								dims=da.dims,
								coords=da.coords  )
	
	return daFiltered


#%% saving video/images

def save_video_to_gif(video, filename='movie.gif',dpi=75, cleanup=True, vmin=0, vmax=int(2**12)):
	import imageio
	
	files=[]
	_plt.ioff()
	print('generating images')
	for i,ti in enumerate(video.t):
# 		print(i,float(ti))
		fig,ax=_plt.subplots()
		video.sel(t=ti).plot(ax=ax,vmin=vmin,vmax=vmax)
		ax.set_title('t=%.9f s'%ti)
		fig.savefig('image_%.10d.png'%i,dpi=dpi)
		files.append('image_%.10d.png'%i)
		_plt.close(fig)
	_plt.ion()
	
	print('compiling gif')
	with imageio.get_writer(filename, mode='I') as writer:
	    for file in files:
	        image = imageio.imread(file)
	        writer.append_data(image)
		
	if cleanup==True:
		print('cleaning up images')
		import os
		for file in files:
			os.remove(file)
			

def save_list_of_figures_to_pdf(figs,filename="output.pdf"):
	"""
	Saves a lit of figures to a multi-page pdf

	Parameters
	----------
	figs : list of figures
		list of figures
	filename : str
		filename of the pdf to be saved.  extension should be .pdf

	References
	----------
	  * https://stackoverflow.com/questions/17788685/python-saving-multiple-figures-into-one-pdf-file

	"""
	
	import matplotlib.backends.backend_pdf
	pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
	for fig in figs: 
	    pdf.savefig( fig )
	pdf.close()
	
	
#%% play video/images
def playVideo(da,interval=200):
	"""
	Play video stored in xarray.dataarray format

	Parameters
	----------
	da : xarray.core.dataarray.DataArray
		3D video with dimensions: ('t', 'y', 'x')  
	interval : int, optional
		Time interval between frames of the video in milliseconds.  E.g. 60 Hz = 0.0167*1000 = 17

	"""
	
	if 'time' in da.dims:
		da=da.rename({'time':'t'})
	if 'x' not in da.dims and 'y' not in da.dims and 't' not in da.dims:
		raise Exception('Dimensions not formatted ocrrectly.  Should be (t,x,y)')
	da=da.transpose('t',...) # 't' must be first.
	
	from matplotlib.animation import FuncAnimation
	
	## initialize plot
	fig, ax = _plt.subplots()
	ax.set_aspect('equal')
	t=da.t.data
	dt=t[1]-t[0]
	n_t=_np.arange(t.shape[0])
	
	## animate
	def animate(i):
		ax.clear()
		da[i,:,:].plot(ax=ax,add_colorbar=False)
		ax.set_title('t = %d/%d,\ndt = %.3e, interval = %d us'%(n_t[i],n_t[-1],dt,interval))
	anim = FuncAnimation(fig, animate, interval=interval, frames=da.t.shape[0])
	_plt.draw()
	_plt.show()
	
	return anim


#%% import video or sequence of images


def add_time_stamp_to_images(df_file_list):
	
	print("work in progress")
	
	for i,file_name in df_file_list.iterrows():
		print(i,file_name)
		file_name=file_name.values[0]
		date_and_time=datetime.datetime.strptime(file_name.split('/')[-1][:-4],'%Y%m%d_%H%M%S')
		dt_string=date_and_time.strftime('%Y %m %d - %H:%M:%S')
		
		
		my_image=Image.open(file_name)
		image_editable = ImageDraw.Draw(my_image)
		font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24, encoding="unic")

		image_editable.text((15,15),dt_string, (255,255,255), font=font)
		if i==0:
			os.chdir('timestamp')
		my_image.save(file_name.split('/')[-1])
		
		
		
def loadSequentialImages(partialFileName):
	"""
	Loads sequential images.
	
	Parameters
	----------
	partialFileName : str
		Directory and filename of the images.  Use wildcards * to indicate
		unspecified text.  Example, partialFileName = 'images/*.tif'
		
	Returns
	-------
	da : xarray.core.dataarray.DataArray
		xarray DataFrame with coordinates: x,y,image
		where image is the image number
	
	Example
	-------
	::
		
		directory='/run/user/1000/keybase/kbfs/private/jwbrooks0/Columbia/Research/thesisWork/hsVideo/101037/'
		partialFileName='%s*.tif'%directory
		da=loadSequentialImages(partialFileName)
		
	References
	----------
	https://rabernat.github.io/research_computing/xarray.html
	"""

	# import libraries
	from PIL import Image
	import glob
	import xarray as xr
	
	# get file names
	inList=glob.glob(partialFileName)
	dfFiles=_pd.DataFrame(inList,columns=['filepaths']).sort_values('filepaths').reset_index(drop=True)
	
	# load files
	img=_np.array(Image.open(dfFiles.filepaths.values[0]))
	if len(img.shape)==2:
		image=_np.zeros((img.shape[0],img.shape[1],dfFiles.shape[0]),dtype=img.dtype)
		for i,(key,val) in enumerate(dfFiles.iterrows()):
			image[:,:,i]=_np.array(Image.open(val.values[0]))
			
			# convert data to xarray DataArray
			da = xr.DataArray(image, 
							  dims=['y', 'x','image'],
			                  coords={
							   'y': _np.arange(0,img.shape[0],dtype=img.dtype),
							   'x': _np.arange(0,img.shape[1],dtype=img.dtype),
							   'image': _np.arange(0,dfFiles.shape[0],dtype=img.dtype)},
							   )
	elif len(img.shape)==3:
		image=_np.zeros((img.shape[0],img.shape[1],img.shape[2],dfFiles.shape[0]),dtype=img.dtype)
		for i,(key,val) in enumerate(dfFiles.iterrows()):
			image[:,:,:,i]=_np.array(Image.open(val.values[0]))
		
			# convert data to xarray DataArray
			da = xr.DataArray(image, 
							  dims=['y', 'x','color','image'],
			                  coords={
							   'y': _np.arange(0,img.shape[0],dtype=img.dtype),
							   'x': _np.arange(0,img.shape[1],dtype=img.dtype),
							   'color': _np.arange(0,img.shape[2],dtype=img.dtype),
							   'image': _np.arange(0,dfFiles.shape[0],dtype=img.dtype)},
							   )
	
	
	return da, dfFiles


def loadSequentialRawImages(	partialFileName,
								dtype=_np.uint16,
								shape=(64,128),
								fps=None,
								color=True,
								plot=False,
								save=False):
	"""
	Loads sequential images.
	
	Parameters
	----------
	partialFileName : str
		Directory and filename of the images.  Use wildcards * to indicate
		unspecified text.  Example, partialFileName = 'images/*.tif'
	dtype : str or numpy data type
		You must specify the data type of the images being loaded.  'uint16' is default.
	shape : tuple with two ints
		These represent the (y,x) resolution of the figure.  NOTE that the order is (y,x).  If you get this backwards, the code will still work, but the image won't look right
	color : bool
		True - assumes that there R,B,G data associated with each image
		False - assumes greyscale.
		
	Returns
	-------
	da : xarray.core.dataarray.DataArray
		xarray DataFrame with coordinates: time,x,y,color
		where time is the image number
	
	Example
	-------
	::
		
		import numpy as np
		
		if False:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\Test1\\test3_goodCaes_15A_30sccm_29p3V'
			shape=(64,128)
			fps=390000
		elif False:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\Test1\\test4_2020_10_12_50mmlens_400kHz_15A_35p5V_15sccm'
			shape=(48,48)
			fps=400000
		elif True:
			directory='C:\\Users\\jwbrooks\\HSVideos\\cathode_testing\\test2\\Test1\\test2_goodCase_15A_20sccm_31p6V'
			shape=(64,128)
			fps=390000
		partialFileName='%s\\Img*.raw'%directory
		
		dtype=np.uint16
		color=True
		save=True
		da=loadSequentialRawImages(partialFileName,shape=shape,dtype=dtype,color=color,save=save,plot=True,fps=fps)
		
	References
	----------
	https://rabernat.github.io/research_computing/xarray.html
	"""
	if fps==None:
		dt=int(1)
	else:
		dt=1.0/fps

	# import libraries
	import glob
	import xarray as xr
	import numpy as np
	
	# get file names
	inList=glob.glob(partialFileName)
	dfFiles=_pd.DataFrame(inList,columns=['filepaths'],dtype=str).sort_values('filepaths').reset_index(drop=True)
	
	# initialize data storage 
	if color==True:
		video=_np.zeros((dfFiles.shape[0],shape[0],shape[1],3),dtype=dtype)
	else:
		video=_np.zeros((dfFiles.shape[0],shape[0],shape[1],1),dtype=dtype)
		
	# step through each image and import it in the data storage
	for i,(key,val) in enumerate(dfFiles.iterrows()):
# 		print(val[0]) # TODO convert to a percent complete printout
		A=_np.fromfile(val[0],dtype=dtype)
		if color==True:
			B=A.reshape((shape[0],shape[1],3))
			video[i,:,:,:]=B[::-1,:,:]
		else:
			B=A.reshape((shape[0],shape[1],1))
			video[i,:,:,0]=B[::-1,:,0] 
	
	# convert to xarray
	t=np.arange(video.shape[0])*dt
	x=np.arange(video.shape[2])
	y=np.arange(video.shape[1])
	if video.shape[3]==1:
		c=['grey']
	elif video.shape[3]==3:
		c=['blue','green','red']
	video_out=xr.DataArray(	video,
							dims=['t','y','x','color'],
							coords={	't':t,
										'x':x,
										'y':y,
										'color':c})
	
	if save==True:
		import pickle as pkl
		pkl.dump(video_out,open(partialFileName.split('*')[0]+'.pkl','wb'))
 	
	if plot==True:
		_plt.figure()
		video_out[0,:,:,0].plot()

	return video_out



def readTiffStack(filename, fps=1):
	
	"""
	Reads a tiff stack and returns a xarray dataarray
	
	Note, this uses a less-than-ideal data loading method because it is more robust.  For example, I have needed to load int12 tiffs in the past, and other image loading functions have no idea what to do with int12. 
	
	References
	----------
	https://stackoverflow.com/questions/37722139/load-a-tiff-stack-in-a-numpy-array-with-python
	
	"""
	
	from PIL import Image
	import xarray as xr
	import numpy as np
	
	data=Image.open(filename)
	
	image=np.zeros(	(data.n_frames,
					 np.shape(data)[0],
					 np.shape(data)[1]),
					dtype=np.array(data).dtype)	
	
	for i in range(data.n_frames):
		data.seek(i)
		image[i,:,:]=np.array(data)
		
	import numpy as _np
	
	time=np.arange(0,image.shape[0])/fps
		
	image = xr.DataArray(image, 
						  dims=['time','y', 'x',],
		                  coords={'x': _np.arange(0,image.shape[2],dtype=np.int16),
						   'y': _np.arange(0,image.shape[1],dtype=np.int16),
						   'time': time},
						   )
	
	return image


#%% SVD code


def svdVideoDecomposition(		X,	
								plot=False,
								saveFigs=False,
								figsName='results.pdf',
								fileRoot='',
								tlim=None,
								nperseg=5):		
	"""
	
	Parameters
	----------
	X : xarray.core.dataarray.DataArray
		Video data in 3D xarray DataArray format with dimensions ('t','y','x').  
		
	References
	----------
	  *  Algrithm is spelled out in Tu2014 - https://www.aimsciences.org/article/doi/10.3934/jcd.2014.1.391
	
	Notes
	-----	
	  *  this method only works if there is NO MASKING on the data 
	
		
	Examples
	--------
	
	Example 1 ::
		
		import os
		from pathlib import Path
		import matplotlib.pyplot as plt; plt.close('all')
		import johnspythonlibrary2 as jpl2
		import pickle as _pkl
# 		filename='C:\\Users\\jwbrooks\\HSVideos\\2011_01\\600V 10A\\tiffs\\600-10_FASTCAM_withdata_1p75MHzGSDAQ_s_tif_unmaskedImages.pkl'
# 		# filename='C:\\Users\\jwbrooks\\HSVideos\\2011_01\\300V 10A\\tiffs\\300-10_FASTCAM_withdata_1p6MHzGSDAQ_s_tif_unmaskedImages.pkl'
# 		baseName,dirName,fileExt,fileRoot=jpl2.OS.processFileName(filename)
# 		
# 		images=_pkl.load(open( filename, "rb" ))
		
	
		fps=175e3
		rootdir=os.environ["HOME"]
		filename=Path(rootdir) / Path('HSVideos/2011_01/600V 10A/tiffs/600-10_FASTCAM_withdata_1p75MHzGSDAQ_s_tif.tif')
		images=jpl2.VideoImage.readTiffStack(filename,fps=fps)
		
							
		X=images[3500:5500,:,:]
		
		topos,chronos,Sigma=svdVideoDecomposition(X,fileRoot=fileRoot,plot=True)
		
		
	"""
	import numpy as _np
	import xarray as xr
	
	## Make sure input is formatted correctly
	if _np.isnan(X.data).any():
		raise Exception('Input signal contains NaN')
	if 'time' in X.dims:
		X=X.rename({'time':'t'})
	if 'x' not in X.dims and 'y' not in X.dims and 't' not in X.dims:
		raise Exception('Dimensions not formatted ocrrectly.  Should be (t,x,y)')
	
	## reshape 3D data to 2D in the desired order = (space, time)
	X=X.stack(z=('y','x')).transpose('z','t')
	
	## svd algorithm
	try:
		U, Sigma, VT = _np.linalg.svd(X, full_matrices=False)
	except:
		try:
			U, Sigma, VT = _np.linalg.svd(X, full_matrices=False)
		except:
			try:
				U, Sigma, VT = _np.linalg.svd(X, full_matrices=False)
			except:
				print('SVD failed after 3 attempts.')
		
	## clip results based on minimum dimension
	n_rank = min(X.shape)
	U = U[:, :n_rank]
	Sigma = Sigma[:n_rank]
	VT = VT[:n_rank, :]
	
	## save results as xarray dataarrays
	energy=xr.DataArray(	Sigma,
						dims=['basisNum'],
						coords={	'basisNum':_np.arange(0,n_rank)})
	
	topos=xr.DataArray(	U,
						dims=['z','basisNum'],
						coords={	'z': X.z,
									'basisNum':_np.arange(0,n_rank)}).unstack('z')
	
	chronos=xr.DataArray(	VT,
							dims=['basisNum','t'],
							coords={ 't':X.t,
									 'basisNum':_np.arange(0,n_rank)})
	
	## optional plots
	if plot==True:
		
		from johnspythonlibrary2.Plot import subTitle, finalizeFigure
		
		_plt.ioff()
		figs=[]
		# Energy plot
		if True:
			fig,ax=_plt.subplots(2,sharex=True)
			energy.plot(ax=ax[0],marker='.')
			ax[0].set_yscale('log')
			ax[0].set_title('Basis energy')
			subTitle(ax[0],'Basis energy')
			ax[0].set_ylabel('Energy (a.u.)')
			
			a=_np.cumsum(energy)/energy.sum()
			ax[1].plot(energy.basisNum,a,marker='.')
	 		# ax[1].set_yscale('log')
			ax[1].set_ylabel('Cumulative sum (normalized)')
			ax[1].set_xlabel('Basis number')
			subTitle(ax[1],'Cumulative sum of mode energy (normalized)')
# 			if saveFig==True:
# 				fig.savefig('%s_basis_energy_HSVideo.png'%(fileRoot))
			figs.append(fig)
				
		# Bases plots
		if True:
			from johnspythonlibrary2.Process.Spectral import fft_df, fft_max_freq, fft, fft_average
			
			for i in range(30):
				fig,ax=_plt.subplots(1,3);
				N=chronos.t.shape[0]//int(nperseg) ## sets the step for the fft_average function
				da_fft=fft_average(chronos[i,:]*_np.sqrt(Sigma[i]),nperseg=N,noverlap=_np.floor(N*0.9).astype(int),plot=False,trimNegFreqs=True,zeroTheZeroFrequency=True,returnAbs=True)
				da_fft['f']=da_fft.f*1e-3
				f_max=fft_max_freq(da_fft)
				(topos.sel(basisNum=i)*_np.sqrt(energy.sel(basisNum=i))).plot(ax=ax[0]) 
				ax[0].set_aspect('equal')
				(chronos.sel(basisNum=i)*_np.sqrt(energy.sel(basisNum=i))).plot(ax=ax[1])
				if type(tlim)!=type(None):
					ax[1].set_xlim(tlim)
				da_fft.plot(ax=ax[2])
	 			# ax[2].set_yscale('log')
				ax[2].set_title(ax[1].get_title()+', peak freq. = %.2f kHz'%f_max)
				ax[2].set_xlabel('kHz')
				ax[2].set_ylabel('Spectral power (au)')
				finalizeFigure(fig,figSize=[12,4])
				subTitle(ax[0], 'spatial mode')
				subTitle(ax[1], 'mode evolution')
				subTitle(ax[2], 'FFT(mode evolution)')
# 				if saveFig==True:
# 					fig.savefig('%s_basis_%d_HSVideo.png'%(fileRoot,i))
				figs.append(fig)
				
		_plt.ion()

		if saveFigs==True:
			save_list_of_figures_to_pdf(figs,figsName)
			_plt.close('all')
			
	return topos,chronos,energy


def svdVideoReconstruction(	topos,
							sigma,
							chronos,
							basesToKeep=_np.arange(0,10),
							plot=False):
	"""
	
		
	import os
	from pathlib import Path
	import matplotlib.pyplot as plt; plt.close('all')
	import johnspythonlibrary2 as jpl2
	import pickle as _pkl
# 		filename='C:\\Users\\jwbrooks\\HSVideos\\2011_01\\600V 10A\\tiffs\\600-10_FASTCAM_withdata_1p75MHzGSDAQ_s_tif_unmaskedImages.pkl'
# 		# filename='C:\\Users\\jwbrooks\\HSVideos\\2011_01\\300V 10A\\tiffs\\300-10_FASTCAM_withdata_1p6MHzGSDAQ_s_tif_unmaskedImages.pkl'
# 		baseName,dirName,fileExt,fileRoot=jpl2.OS.processFileName(filename)
# 		
# 		images=_pkl.load(open( filename, "rb" ))
	

	fps=175e3
	rootdir=os.environ["HOME"]
	filename=Path(rootdir) / Path('HSVideos/2011_01/600V 10A/tiffs/600-10_FASTCAM_withdata_1p75MHzGSDAQ_s_tif.tif')
	images=jpl2.VideoImage.readTiffStack(filename,fps=fps)
	
						
	X=images[3950:5200,:,:]
	
	topos,chronos,sigma=svdVideoDecomposition(X,plot=True)
	
	
	basesToKeep=[2,3]	
	svd_m3_recon=svdVideoReconstruction(topos,sigma,chronos,basesToKeep,plot=True)
	
	
	"""
	
	## reshape topos into the correct 2D dimensions
	topos=topos.stack(z=('y','x')).transpose('z','basisNum')
	
	## Remove undesired bases
	topos_short=topos.sel(basisNum=basesToKeep).data
	sigma_short=sigma.sel(basisNum=basesToKeep)
	sigma_short=_np.diag(sigma_short)
	chronos_short=chronos.sel(basisNum=basesToKeep).data
	
	## Perform reconstruction
	def svd_recon(U,S,VT):
		return _np.matmul(_np.matmul(U,S),VT)
	recon = svd_recon(topos_short,sigma_short,chronos_short)
	
	## Concert results to xarray dataarray
	recon=_xr.DataArray(	recon,
							dims=['z','t'],
							coords={'z':topos.z,
									't':chronos.t}).unstack('z')
	
	## Optional plot
	if plot == True:
		dt=2 # time step between images
		fig,ax=_plt.subplots(4,4)
		ij=0
		for i in _np.arange(0,4):
			for j in _np.arange(0,4):
				recon[ij,:,:].plot(ax=ax[i,j])
				ax[i,j].set_xlabel('')
				ax[i,j].set_ylabel('')
				ax[i,j].set_xticklabels([''])
				ax[i,j].set_yticklabels([''])
				ax[i,j].set_title('%.3fms'%(recon[ij,:,:].t.data*1e3))
				ij+=dt
	
	return recon