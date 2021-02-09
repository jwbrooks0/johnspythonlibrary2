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


#TODO this function needs to be checked and cleaned up
class langmuirProbe:
    """
    Langmuir probe analysis.  Returns temperature and density of a plasma 
    with a provided I-V profile. 
    The code fits an exponential fit between the floating potential and the 
    plasma potential in order provide the temperature and density.  
    
    Parameters
    ----------
    V : np.array() of floats
        Voltage of probe.  Units in volts.
    I : np.array() of floats
        Current of probe.  Units in amps.
    expRegionMinVoltage : float or NoneType
        Minimum potential to be used in the exponential fit
        if None, you will given a plot of the I-V profile and asked for the value
    expRegionMaxVoltage : float or NoneType
        Maximum potential to be used in the exponential fit
        if None, you will given a plot of the I-V profile and asked for the value
    ionSatRegionMaxVoltage : float or NoneType
        Maximum voltage value to consider for the ion saturation region  
        if None, you will given a plot of the I-V profile and asked for the value
    area : float
        probe area in cubic meters.  
        HBT-EP's BP: 0.75 inch diameter, half sphere ->  
                     4*pi*(0.0254*.75/2)^2/2 = 0.000580644 m^3
    plot : bool
        True = plots final fit 
        False = does not
    expFitGuess : tuple of three floats
        provide a guess for the exponential fit
        expFitGuess = (a,b,c) 
        where the function is a*_np.exp(b*x)+c
        a = the amplitude
        b = exp const
        c = y-offset
        expFitGuess = (6, 0.05, -5) by default 
        
    Attributes
    ----------
    #TODO(John) add attributes
    
    Notes
    -----
    Using the exponential offset from the current data gives an uncomfortably 
    large density value.  This is probably not the best way to do this.  
        
    """
    
    
    def __init__(self, V, I, expRegionMinVoltage=None, expRegionMaxVoltage=None, ionSatRegionMaxVoltage=None,
                 area=0.000580644,
                 plot=False, expFitGuess=(6, 20, -5)):
        ## physical constants
#        eV=1.60218e-19;
#        mi=1.6737236 * 10**(-27) * 2
#        q=1.6e-19
        
        # parameters
        self.probeArea=area
        self.expRegionMinVoltage=expRegionMinVoltage
        self.expRegionMaxVoltage=expRegionMaxVoltage
        self.ionSatRegionMaxVoltage=ionSatRegionMaxVoltage
        
        # ensure V and I are arrays
        self.V=_np.array(V); 
        self.I=_np.array(I)
        
        # sort V in ascending order
        i=_np.argsort(self.V)
        self.V=self.V[i]
        self.I=self.I[i]
        
        # initialize plot
        p1=_plt.plot()
        p1.addTrace(xData=self.V,yData=self.I,marker='.',linestyle='',
                         yLegendLabel='raw data')
        
        # if expRegionMinVoltage or expRegionMaxVoltage were not specified, the code will plot the I-V
        # profile and ask that you provide the floating and/or plasma 
        # potential.  These values are used as the lower and upper limits for
        # the exponential fit
        if expRegionMinVoltage==None or expRegionMaxVoltage==None or ionSatRegionMaxVoltage==None:
            p1.plot()
            _plt.show()
            _plt.pause(1.0) # a pause is required for the plot to show correctly
            if expRegionMinVoltage==None:
                self.expRegionMinVoltage=float(input("Please provide the approximate lower voltage (units in volts) limit to be used in the exp fit by looking at the I-V plot:  "))
            if expRegionMaxVoltage==None:
                self.expRegionMaxVoltage=float(input("Please provide the approximate upper voltage (units in volts) limit to be used in the exp fit by looking at the I-V plot:  "))
            if ionSatRegionMaxVoltage==None:
                self.ionSatRegionMaxVoltage=float(input("Please provide the approximate maximum voltage (units in volts) limit for the ion saturation region by looking at the I-V plot:  "))
        
        # exp. curve fit setup
        from scipy.optimize import curve_fit

        # perform exp curve fit
        popt, pcov = curve_fit(self._exponenial_func, V, I, p0=expFitGuess)
        self.expFitParameters=popt
#        print popt
        
        # temperature
        self.temperatureInEV=self.calcTempInEV(popt[1])#q*popt[1]/eV
        print("Temperature = " + str(self.temperatureInEV) + ' eV')
        
        # density calculation from the exp fit offset
        self.densityFromExpFitOffset=self.calcDensity(popt[2],
                                                      probeArea=self.probeArea,
                                                      temperatureInEV=self.temperatureInEV)
        print("Density from the exp. fit offset current = " + str(self.densityFromExpFitOffset) + ' m^3')
        
        # density calculation from averaging the values in the ion sat region
        i=self.V<self.ionSatRegionMaxVoltage
        aveIonSatCurrent=_np.average(self.I[i])
        print("Average Current in the ion sat. region = " + str(aveIonSatCurrent))
        self.densityFromAveIonSatRegion=self.calcDensity(aveIonSatCurrent,
                                                      probeArea=self.probeArea,
                                                      temperatureInEV=self.temperatureInEV)
        print("Density from the average current in the ion sat. region = " + str(self.densityFromAveIonSatRegion) + ' m^3')
        
        # optional plot
        if plot==True:
            self.plot()
            
            
    def calcTempInEV(self, expFitCoeffWithVoltUnits):
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
        
        
    def calcDensity(self, ionSatCurrent, probeArea, temperatureInEV):
        """
        Calulates density
        
        Parameters
        ----------
        expFitCoeffWithVoltUnits : float
            
        """
        # constants
        eV=1.60218e-19;
        q=1.6e-19
        mi=1.6737236 * 10**(-27) * 2
        
        # thermal velocity
        vth=_np.sqrt(2*temperatureInEV*eV/mi)
        
        # density
        return 4*_np.abs(ionSatCurrent)/q/probeArea/vth
        
        
    def plot(self):
        """ plot raw data and exp fit """
        
        # exp fit
        xFit=_np.arange(self.expRegionMinVoltage,self.expRegionMaxVoltage,0.1)
        yFit=self._exponenial_func(xFit,self.expFitParameters[0],
                                  self.expFitParameters[1],
                                  self.expFitParameters[2])
        
        # extrapolated exp fit 
        xFitExtrap=_np.arange(_np.min(self.V),_np.max(self.V),0.1)
        yFitExtrap=self._exponenial_func(xFitExtrap,self.expFitParameters[0],
                                  self.expFitParameters[1],
                                  self.expFitParameters[2])
        
        # generate plot
        p1=_plt.plot(title='I-V Profile',xLabel='Probe Voltage [V]',
                      yLabel='Probe Current [A]')
        p1.addTrace(xData=self.V,yData=self.I,marker='.',linestyle='',
                         yLegendLabel='Raw data',alpha=0.5)
        p1.addTrace(xData=xFit,yData=yFit, yLegendLabel='Exp fit')
        p1.addTrace(xData=xFitExtrap,yData=yFitExtrap, 
                    yLegendLabel='Extrapolated exp fit',linestyle=':')
        p1.plot()
        
        return p1
        
        
    def _exponenial_func(self,x, a, b, c):
        """ exponential function """
        return a*_np.exp(x/b)+c