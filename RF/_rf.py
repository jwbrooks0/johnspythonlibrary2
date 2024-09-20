

import pandas as _pd
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Plot import legend_above, \
	finalizeSubplot, finalizeFigure


# %% plotting subfunctions

## TODO this lib needs an overhaul
## most of the relevant functionality has been moved to pip_tools

def _finalize_complex_plot(fig,ax,title='', ylabel='Ohm'):
	finalizeSubplot(ax[0,0],
					 subtitle='Abs',
					 title=title,
					 ylabel=ylabel,
					 legendOn=False
					 )
	finalizeSubplot(ax[1,0],
					 xlabel='Frequency (MHz)',
					 subtitle='Phase',
					 ylabel='deg.',
					 legendOn=False,
					 ylim=[-180,180]
					 )
	finalizeSubplot(ax[0,1],
					 subtitle='Real',
					 ylabel=ylabel,
					 )
	finalizeSubplot(ax[1,1],
					 xlabel='Frequency',
					 subtitle='Imag(Z)',
					 ylabel=ylabel,
					 legendOn=False
					 )
	legend_above(ax[0,1])
	
	finalizeFigure(fig, figSize=[9,5])
	
	
def plot_complex(Z,ax=None,fig=None, label='Z',linestyle='-',color='k',yscale='symlog', linewidth=1.25, marker='', pdiff=[], ylabel='Ohm'):
	if type(Z) == _pd.core.frame.DataFrame:
		Z=Z.TANK.copy()
		f=Z.index.values
	elif type(Z) == _xr.core.dataarray.DataArray:
		Z=Z.copy()
		if 'f' in Z.dims:
			f=Z.f.data
		else:
			f=Z.Frequency.data
		
	if type(ax)==type(None):
		fig,ax=_plt.subplots(2,2, sharex=True)
		
	ax[0,0].plot(f*1e-6,_np.abs(Z),label='%s'%label,linestyle=linestyle,color=color, linewidth=linewidth, marker=marker)
	if len(pdiff)==0:
		 ax[1,0].plot(f*1e-6,180/_np.pi*_np.arctan2(_np.imag(Z),_np.real(Z)),label='%s'%label,linestyle=linestyle,color=color, linewidth=linewidth, marker=marker)
	else:
		ax[1,0].plot(pdiff.f*1e-6,180/_np.pi*pdiff.data,label='%s'%label,linestyle=linestyle,color=color, linewidth=linewidth, marker=marker)
	
	ax[0,1].plot(f*1e-6,_np.real(Z),label='%s'%label,linestyle=linestyle,color=color, linewidth=linewidth, marker=marker)
	ax[1,1].plot(f*1e-6,_np.imag(Z),label='%s'%label,linestyle=linestyle,color=color, linewidth=linewidth, marker=marker)
	ax[0,0].set_yscale(yscale)
	ax[0,1].set_yscale(yscale)
	ax[1,1].set_yscale(yscale)
	for i in [0,1]:
		ax[i,0].plot(f*1e-6,_np.zeros(len(f)),linestyle=':',c='grey')

	_finalize_complex_plot(ax[0,0].get_figure(),ax, ylabel=ylabel)
	
	return fig,ax


# #%% Load/Save data

# def save_Z_to_csv(Z, filename):
# 	
# 	Z2=Z.to_pandas()
# 	Z2.name=Z.name
# 	Z2.to_csv(filename)
# 	
# def load_Z_from_csv(filename):
# 	
# 	try:
# 		Z=_pd.read_csv(filename)#.set_index('f')
# 		Z=_pd.DataFrame(Z.iloc[:,1].values,index=Z.iloc[:,0].values)
# 	except:
# 		Z_temp=_pd.read_csv(filename, skiprows=2).set_index('Frequency')
# 		Z=_pd.DataFrame( _np.real(Z_temp.iloc[:,0]) + _np.imag(Z_temp.iloc[:,1]), index=Z_temp.index)
# 		
# 	Z=_xr.DataArray(Z.values.transpose()[0].astype(complex), dims='f', coords=[Z.index.values])

# 	return Z
	