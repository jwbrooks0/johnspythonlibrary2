# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:20:38 2021

@author: jwbrooks
"""


##################################
# %% Import libraries

import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr

from johnspythonlibrary2.Constants import ep_0, e, amu, m_Ar, pi, k_b, eV, pi, m_e, m_Xe
from johnspythonlibrary2.Process.Fit import polyFitData

##################################
# %% Plasma parameters
def debye_length(
        density, 
        temperature_in_eV,
        ):
    """
    
    References
    ----------
    * Merlino, 2007.  Eq. 4. https://aapt.scitation.org/doi/10.1119/1.2772282
    """
    
    # TODO something here is wrong...
    
    return _np.sqrt(ep_0 * temperature_in_eV / ( density * e) )


def plasma_frequency(
        density,  # untis in m^-3
        mass = m_e, # defaults to the electron-plasma frequency
        # mass = m_Ar,
        # plot=False,
        # return_xr=False,
        ):
    """
    
    Example
    -------
    Example 1::
        
        density = 10**_np.arange(13,19,0.01)
        plasma_frequency_from_density(density)
    """
    freq = _np.sqrt(density * e**2 / (mass * ep_0)) * 1 / (2 * pi)
    # freq_xr = _xr.DataArray(
    #     freq,
    #     dims='n',
    #     coords=[density],
    #     attrs={'long_name':'Frequency',
    #               'units':'Hz'})
    # freq_xr.n.attrs={'long_name':'Density', 'units':r'$m^{-3}$'}
    
    # if plot is True:
    #     fig, ax = _plt.subplots()
    #     freq_xr.plot(ax=ax)
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.grid()
        
    # if return_xr is True:
    #     return freq_xr
    # else: 
    return freq


def density(
        frequency, 
        mass=m_e, 
        plot=False,
        ):
    """
    
    Example
    -------
    Example 1::
        
        frequency = 10**_np.arange(7,12,0.01)
        mass = m_Ar
        density=plasma_density(frequency, mass, plot=True)
    """
    density = (2*pi*frequency)**2 * mass * ep_0 / e**2

    return density


def cyclotron_frequency(B, m):
    return e * B / (2*pi*m)


def lower_hybrid_frequency( m_ion, B, n0, plot=False):
    """ Bellan: eq. 6.40 """
    w_pe=plasma_frequency(density=n0, mass=m_e)
    w_pi=plasma_frequency(density=n0, mass=m_ion)
    
    w_ce=cyclotron_frequency(B=B, m=m_e) 
    w_ci=cyclotron_frequency(B=B, m=m_ion)
    if type(n0)==_np.ndarray:
        w_ce *= _np.ones(len(n0))
        w_ci *= _np.ones(len(n0))
    else:
        w_pe *= _np.ones(len(B))
        w_pi *= _np.ones(len(B))
        
    w_lh = _np.sqrt(w_ci**2 + w_pi**2 / (1 + w_pe**2 / w_ce**2))
    
    if plot is True:
        fig, ax =_plt.subplots()
        if type(n0) == _np.ndarray:
            x = n0
        else:
            x = B
        ax.plot(x, w_lh, label='w_lh')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    return w_lh



##################################
# %% Probe analysis

## My Langmuir probe analysis has been moved to a separate library
## Same as my double langmuir probe
    
