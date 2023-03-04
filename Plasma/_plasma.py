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
def debye_length(density, temperature_in_eV):
	"""
	
	References
	----------
	* Merlino, 2007.  Eq. 4. https://aapt.scitation.org/doi/10.1119/1.2772282
	"""
	
	# TODO something here is wrong...
	
	return _np.sqrt(ep_0 * temperature_in_eV / ( density * e) )


def plasma_frequency_from_density(density,  plot = False):
	"""
	
	Example
	-------
	Example 1::
		
		density = 10**_np.arange(13,19,0.01)
		plasma_frequency_from_density(density, plot=True)
	"""
	mass = m_e
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
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.grid()
		
	return frequency



def density_from_frequency(frequency, mass=m_e, plot = False):
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


def plasma_frequency( n0, m=m_e):
	return _np.sqrt(n0 * e**2 / (m * ep_0)) * 1/(2*pi)


def cyclotron_frequency( B, m):
	return e * B / (2*pi*m)


def lower_hybrid_frequency( m_ion, B, n0, plot=False):
	""" Bellan: eq. 6.40 """
	w_pe=plasma_frequency(n0=n0, m=m_e)
	w_pi=plasma_frequency(n0=n0, m=m_ion)
	
	w_ce=cyclotron_frequency(B=B, m=m_e) 
	w_ci=cyclotron_frequency(B=B, m=m_ion)
	if type(n0)==_np.ndarray:
		w_ce *= _np.ones(len(n0))
		w_ci *= _np.ones(len(n0))
	else:
		w_pe *= _np.ones(len(B))
		w_pi *= _np.ones(len(B))
		
		
	w_lh = _np.sqrt( w_ci**2 + w_pi**2/(1+w_pe**2/w_ce**2))
	
	if plot==True:
		fig,ax=_plt.subplots()
		if type(n0)==_np.ndarray:
			x=n0
		else:
			x=B
		ax.plot(x,w_lh,label='w_lh')
		ax.legend()
		ax.set_yscale('log')
		ax.set_xscale('log')
	
	return w_lh



##################################
# %% Probe analysis

def IV_sweep_analysis_v3(da, 
						 probe_area=3.1669e-5,  
						 expFitGuess=(6, 20, -5), 
						 mass = m_Ar, 
						 cold_plasma_assumption=True,
						 plot=True, 
						 ): 

	
	def density_from_isat_and_temp(ionSatCurrent, temperatureInEV, probe_area = probe_area, cold_plasma_assumption = cold_plasma_assumption, mass = mass):
		"""
		Calulates density
		
		References
		----------
		* Merlino, 2007. https://aapt.scitation.org/doi/10.1119/1.2772282
		"""
	# 		mi=1.6737236 * 10**(-27) * 39.948 # argon
		
		if cold_plasma_assumption==False:
			# v_thermal=_np.sqrt(2*temperatureInEV*eV/mass)
			v_thermal=_np.sqrt(8 * temperatureInEV * eV/ (mass * pi ))
			# Merlino, eq. 2
			density= 4*_np.abs(ionSatCurrent)/e/probe_area/v_thermal
		else:
			v_bohm=_np.sqrt( temperatureInEV * eV / mass)
			# Merlino, eq. 3
			density=_np.abs(ionSatCurrent)/(0.6 * e * probe_area * v_bohm)
			
		return density
	
		
	def calcTempInEV(expFitCoeffWithVoltUnits):
		"""
		Calulates temperature from langmuir exp fit
		
		Parameters
		----------
		expFitCoeffWithVoltUnits : float
			
		"""
		# constants
		eV=1.60218e-19;
		q=1.6e-19
		
		# temperature in eV
		return q*expFitCoeffWithVoltUnits/eV
		
	def _exponenial_func(x, a, b, c):
		""" exponential function """
		return a*_np.exp(x/b)+c
	
	# determine floating potential
	
	def find_V_float(da,plot=False):
	
		from scipy.interpolate import interp1d
		f2 = interp1d(da.data, da.V.data, kind='cubic')
		V_float=f2(0)
		
		if plot==True:
			
			fig, ax=_plt.subplots()
			da.plot(ax=ax, marker='x',linestyle='')
			I_fit=_np.linspace(da.data.min(),da.data.max(),1000)
			V_fit=f2(I_fit)
			ax.plot(V_fit,I_fit,label='cubic spline')
			ax.legend()
			
		return V_float
	
	V_float = find_V_float(da)
	exp_fit_bounds=[V_float-1, V_float+10]  #TODO this upperlimit is a hack.  Instead, develop a code to autodetermine the V_plasma and use that (or midpoint between V_f and V_p) as the upper limit.  
	
	# subtract ion saturation linear fit from data
	da_isat=da[da.V<V_float]
	da_isat=da_isat[da_isat.V<da_isat.V.data.mean()]
	_, i_sat_fit, _, _=polyFitData(da_isat, order=1,plot=False)
	da_e_current=da.copy()-i_sat_fit(da.V.data)
	
	
	# exp. curve fit setup
	from scipy.optimize import curve_fit
	
	dIe_dV = _xr.DataArray( _np.gradient(da_e_current)/_np.gradient(da_e_current.V), dims='V', coords=[da_e_current.V])
# 	dIe2_dV2 = _xr.DataArray( _np.gradient(dIe_dV)/_np.gradient(dIe_dV.V), dims='V', coords=[da_e_current.V])
	if False:
		fig,ax=_plt.subplots()
		dIe_dV.plot(ax=ax, yscale='log')

	## perform exp curve fit
	# fit exp
	da_exp = da_e_current[(da_e_current.V<=exp_fit_bounds[1]) & (da_e_current.V>=exp_fit_bounds[0])]
	expFitParameters, _ = curve_fit(_exponenial_func, da_exp.V.data, da_exp.data, p0=expFitGuess)
	
	# temperature
	temperature_in_eV=calcTempInEV(expFitParameters[1])#q*popt[1]/eV
	
		
	print("Temperature = %.3f eV " % temperature_in_eV)
	
	# density calculation from averaging the values in the ion sat region
# 	da_isat=da[(da.V<=i_sat_bounds[1]) & (da.V>=i_sat_bounds[0])]
	aveIonSatCurrent=_np.average(da_isat)
	print("Average I_sat = %.3e A"%aveIonSatCurrent)
	density=density_from_isat_and_temp(aveIonSatCurrent,
						temperature_in_eV)
	print("Density from the average current in the ion sat. region = %.3e m^2 "%density)
	
	debye=debye_length(density, temperature_in_eV)
	print('Debye length = %.2e m'%(debye))
	
	radius = _np.sqrt(probe_area/_np.pi)
	print('r = %.2e m'%radius)
	
	if plot==True:
		
		fig,ax=_plt.subplots()
		_np.abs(da).plot(ax=ax, label='Raw', marker='x', linestyle='')
		
		_np.abs(da_e_current).plot(ax=ax, label='Electron current', marker='+', linestyle='')
		
		v_fit=_np.arange(exp_fit_bounds[0],exp_fit_bounds[1],0.1)
		da_fit=_xr.DataArray ( _exponenial_func(v_fit, expFitParameters[0], expFitParameters[1], expFitParameters[2]),
							   dims='V',
							   coords=[v_fit])
		
		_np.abs(da_fit).plot(ax=ax, label='exp. fit', linewidth=2)
		
		v_isat=da_isat.V.data
		da_isat_fit = _xr.DataArray(i_sat_fit(v_isat),
								 dims='V',
								 coords=[v_isat])
		print(da_isat_fit)
		
		_np.abs(da_isat_fit).plot(ax=ax, label='Linear I_sat fit', linewidth=2)
		
		ax.legend()
		ax.set_title('Temperature = %.1f eV,     Density = %.1e %s, \n Debye length = %.1e m, r = %.1e m, V_float=%.3f'%(temperature_in_eV,density,r'm$^{-3}$',debye, radius, V_float))
		ax.set_yscale('log')
		ax.set_xlabel('Volts [V]')
		ax.set_ylabel('|Current| [A]')
		
	return temperature_in_eV, density, V_float, da #, da_e_current
	
	
