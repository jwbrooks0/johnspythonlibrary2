
import audio2numpy as _a2n
import numpy as _np
import xarray as _xr
from johnspythonlibrary2.Process.Spectral import stft
import matplotlib.pyplot as _plt
import pandas as _pd

def read_mp3(filepath, plot=False, 
		nperseg=5000,
		noverlap=0):
	filepath='C:\\Users\\jwbrooks\\Downloads\\2021_03_25_20_56_34.mp3'
	audio, f_s=_a2n.open_audio(filepath)
	
	t=_np.arange(len(audio))/f_s
	
	if len(audio.shape) == 1:
		audio = _xr.DataArray( audio, dims='t', coords=[t])
	else:
		audio = _xr.DataArray( audio[:,0], dims='t', coords=[t])
	
	if plot==True:
		fig,ax=_plt.subplots()
		audio.plot(ax=ax)
		stft_results=stft(audio, numberSamplesPerSegment=nperseg, numberSamplesToOverlap=noverlap,plot=False, logScale=True)
		fig,ax=_plt.subplots()
		_np.abs(stft_results).plot(ax=ax)
		
	return audio
	
def notes_to_frequency():
	return _pd.read_csv('notes_to_frequency.csv').set_index('count').dropna(axis=1)