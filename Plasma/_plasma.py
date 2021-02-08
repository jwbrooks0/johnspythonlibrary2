# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:20:38 2021

@author: jwbrooks
"""

import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr

from johnspythonlibrary2.Constants import ep_0, e, amu, m_Ar, pi


def plasma_frequency(density, mass, plot = False):
	"""
	
	Example
	-------
	Example 1::
		
		density = 10**_np.arange(12,21,0.01)
		mass = m_Ar
		plasma_frequency(density, mass, plot=True)
	"""
	frequency = _xr.DataArray(_np.sqrt(density * e**2 / (mass * ep_0)) * 1/(2*pi),
							  dims='n',
							  coords=[density],
							  attrs={'long_name':'Frequency',
										'units':'Hz'})
	frequency.n.attrs={'long_name':'Density',
										'units':r'$m^{-3}$'}
	if plot == True:
		fig,ax=_plt.subplots()
		frequency.plot(ax=ax)
		ax.set_title('mass = %.3f AMU'%(mass/amu))
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.grid()
		
	return frequency



def plasma_density(frequency, mass, plot = False):
	"""
	
	Example
	-------
	Example 1::
		
		frequency = 10**_np.arange(7,12,0.01)
		mass = m_Ar
		plasma_density(frequency, mass, plot=True)
	"""
	density = _xr.DataArray( (2*pi*frequency)**2 * mass * ep_0 / e**2,
							  dims='f',
							  coords=[frequency],
							  attrs={'long_name':'Density',
										'units':r'$m^{-3}$'})
	density.f.attrs={'long_name':'Frequency',
										'units':'Hz'}
	if plot == True:
		fig,ax=_plt.subplots()
		density.plot(ax=ax)
		ax.set_title('mass = %.3f AMU'%(mass/amu))
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.grid()

	return density
