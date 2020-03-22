
import numpy as _np
#import pandas as _pd
import matplotlib.pyplot as _plt
#from johnspythonlibrary2 import Plot as _plot
from johnspythonlibrary2.Process.Misc import findNearest



def chirp(t,tStartStop=[0,1],fStartStop=[1e3,1e4],phi=270,plot=False,method='logarithmic'):
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
	
	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(t,y)
		
	return y



