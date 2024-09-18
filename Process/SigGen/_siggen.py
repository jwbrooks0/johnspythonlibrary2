
import numpy as _np
import matplotlib.pyplot as _plt
from johnspythonlibrary2.Process.Misc import findNearest
import xarray as _xr

# from scipy.signal import gaussian as _gaussian


# %% Noise

def gaussianNoise(t, mean=0, stdDev=1, plot=False):
	"""
	Produces Gaussian noise based on the standard deviation
	This is a wrapper for the numpy.random.normal function

	Parameters
	----------
	shape : tuple of ints
		Shape of desired noise; such as generated from np.shape()
	mean : float
		Offset value. The default is 0.
	stdDev : float
		"Amplitude" (standard deviation) of the noise. The default is 1.
	plot : bool, optional
		Optional plot of results

	Returns
	-------
	noise : numpy array
		Array containing the generated noise.  
		
	Examples
	--------
	
	Example 1::
		
		t=np.arange(0,10e-3,2e-6)
		y1=np.sin(2*np.pi*t*1e3)
		y2=y1+gaussianNoise(t,mean=0,stdDev=0.5)
		fig,ax=plt.subplots()
		ax.plot(t,y2,label='signal with noise')
		ax.plot(t,y1,label='signal without noise')
		ax.legend()

	"""
	
	noise = _xr.DataArray(	_np.random.normal(mean, stdDev, t.shape),
							dims=['t'],
							coords={'t': t})
	
	if plot==True:
		fig, ax = _plt.subplots()
		ax.plot(noise)
		
	return noise



def uniform_noise(t, min_to_max_amp=1, mean=0, plot=False):
	"""
	Produces random noise using a uniform distribution.  
	Uses the np.random.rand() function.

	Parameters
	----------
	shape : tuple of ints
		Shape of desired noise; such as generated from np.shape()
	min_to_max_am : float
		"Amplitude" of the noise. The default is 1.
	mean : float
		Offset value. The default is 0.
	plot : bool, optional
		Optional plot of results

	Returns
	-------
	noise : numpy array
		Array containing the generated noise.  
		
	Examples
	--------
	
	Example 1::
		
		t=np.arange(0,10e-3,2e-6)
		y1=np.sin(2*np.pi*t*1e3)
		noise = uniform_noise(t, mean=0, min_to_max_amp=0.5)
		y2=y1+ noise
		fig,ax=plt.subplots()
		noise.plot(ax=ax, label='noise')
		ax.plot(t,y2,label='signal with noise')
		ax.plot(t,y1,label='signal without noise')
		ax.legend()

	"""
	
	noise = _xr.DataArray(	(_np.random.rand(len(t)) - 0.5) * min_to_max_amp + mean,
							dims=['t'],
							coords={'t': t})
	
	if plot==True:
		fig, ax = _plt.subplots()
		ax.plot(noise)
		
	return noise


# %% Pulses

def gaussian_pulse(sigma, t, amplitude=1, plot=False):
	"""
	Basic gaussian pulse with units in time
	std_time is the standard deviation of the pulse with units of t
	
	Example
	-------
	
	Example 1::
		
		dt=1e-9
		t=np.arange(0,0.001+dt/2,dt)
		t-=t.mean()
		std_time=1e-4
		gaussian_pulse(std_time, t=t, plot=True)
		
	References
	----------
	  * http://www.cse.psu.edu/~rtc12/CSE486/lecture11_6pp.pdf
	"""
	out= _xr.DataArray( _np.exp(-t**2 / (2 * sigma**2)), dims='t', coords=[t])
	if plot == True:
		out.plot()
	return out


def gaussian_1st_deriv(sigma, t, amplitude=1, plot=False):
	"""
	Basic gaussian pulse with units in time
	std_time is the standard deviation of the pulse with units of t
	
	Example
	-------
	
	Example 1::
		
		dt=1e-9
		t=np.arange(0,0.001+dt/2,dt)
		t-=t.mean()
		std_time=1e-4
		s=gaussian_1st_deriv(sigma = std_time, t=t, plot=True)
		
	References
	----------
	  * http://www.cse.psu.edu/~rtc12/CSE486/lecture11_6pp.pdf
	"""
	# dt=t[1]-t[0]
	out= _xr.DataArray( -t / (2 * sigma**2) * _np.exp(-t**2 / (2 * sigma**2)) * amplitude * sigma * 3.298, dims='t', coords=[t])
	if plot == True:
		fig,ax=_plt.subplots()
		out.plot(ax=ax)
	return out




# %% Basic waveforms


def squareWave(	freq,
				time,
				duty_cycle=0.5,
				plot=False):
	"""
	
	Examples
	--------
	Example 1::
		
		freq=1e2
		dt=1e-5
		time=np.arange(10000)*dt
		
		fig,ax=plt.subplots()
		ax.plot(	squareWave(freq,time))
	"""
	from scipy import signal as sg
	
	y = _xr.DataArray(	sg.square(2 * _np.pi * freq * time, duty=duty_cycle),
						dims=['t'],
						coords={'t':time})
	
	# optional plot
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(y)
		
	return y


def chirp(	t,
			tStartStop=[0, 1],
			fStartStop=[1e3, 1e4],
			phi=270,
			plot=False,
			method='logarithmic'):
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
	iStart=findNearest(t, tStartStop[0])
	iStop=findNearest(t, tStartStop[1])
	
	# create chirp signal using scipy function.  times are (temporarily) shifted such that the phase offset, phi, still makes sense
	y=_np.zeros(len(t))
	y[iStart:iStop] = chirp(t[iStart:iStop] - tStartStop[0], fStartStop[0], tStartStop[1] - tStartStop[0], fStartStop[1], method, phi=phi)
	y= _xr.DataArray(	y,
						dims=['t'],
						coords={'t':t})

	# optional plot
	if plot==True:
		fig, ax = _plt.subplots()
		y.plot(ax=ax)
		
	return y



# %% More complicated functions

def tentMap(N=1000, plot=False, ICs={'x0': _np.sqrt(2) / 2.0}):
	""" 
	generate fake data using a tentmap 
	
	x=tentMap(1000,plot=True)
	
	References
	----------
	* https://en.wikipedia.org/wiki/Tent_map
	"""
	t = _np.arange(0, N+1)
	x = _np.zeros(t.shape, dtype=_np.float64)
	x[0] = ICs['x0']
	def tentMap(x, mu=_np.float64(2 - 1e-15)):
		for i in range(1, x.shape[0]):
			if x[i-1] < 0.5:
				x[i] = mu * x[i - 1]
			elif x[i - 1] >= 0.5:
				x[i] = mu * (1 - x[i - 1])
			else:
				raise Exception('error')
		return x
	
	x = tentMap(x)
	
	# the actual data of interest is the first difference of x
	xDelta=_xr.DataArray(	x[1:] - x[0: -1],
							dims=['t'],
							coords={'t': t[1:]},
							attrs={'units': "au",
								 'standard_name': 'Amplitude'})
	xDelta.t.attrs = {'units': 'au'}
	
	if plot is True:
		fig, ax = _plt.subplots(2, sharex=True)
		ax[0].plot(t, x, linestyle='-', marker='.', linewidth=0.5, markersize=3, label='x')
		xDelta.plot(ax=ax[1], linestyle='-', marker='.', linewidth=0.5, markersize=3, label='x derivative')
		ax[0].legend()
		ax[1].legend()
		
	return xDelta


def saved_lorentzAttractor(	N=2000,
							dt=0.05,
							ICs={	'x0': -9.38131377,
									'y0': -8.42655716 , 
									'z0': 29.30738524},
							plot=False,
							removeMean=False,
							normalize=False,
							removeFirstNPoints=500,
							args={	'sigma': 10.0,
									'b': 8.0 / 3.0,
									'r': 28.0}):
	
	filename='lorentz_attractor_N_%d_dt_%.6f_x0_%.3f_y0_%.3f_z0_%.3f.NetCDF' % (N, dt, ICs['x0'], ICs['y0'], ICs['z0'])
	
	try:
		ds = _xr.open_dataset(filename)
		print('loaded previously generated dataset')
	except:
		print('creating dataset')
		ds = lorentzAttractor(	N=N, 
								dt=dt, 
								ICs=ICs, 
								plot=plot, 
								removeMean=removeMean,
								normalize=normalize,
								removeFirstNPoints=removeFirstNPoints,
								args=args)
		
		ds.to_netcdf(filename)
		ds = _xr.open_dataset(filename)
	
	return ds



def lorentz96_fast_slow(	I=25, 
							J=15, 
							F=8, 
							t_final=1000.0, 
							dt=0.001, 
							params=dict(h=1, b=10, c=10),
							plot=False,
							):
	
	"""
	Parameters
	----------
	I : int
		Number of x equations
	J : int
		Number of y equations
	F : int
		Forcing value
	t_final : float
		Duration of simulation
	dt : float
		Time step
	params : dict
		Dictionary of misc function parameters
	plot : bool
		Optional plot
		
	References
	----------
	 * https://cdanfort.w3.uvm.edu/research/2014-frank-ijbc.pdf
	"""
	print("Work in progress")
	
	# temporarily specify all inputs to help with debugging
	if True:
		I = 25
		J = 15
		F = 8
		params = dict(h=1, b=10, c=10)
		t_final = 1000.0
		dt = 0.001
	
	# assign parameter values
	h = params['h']
	b = params['b']
	c = params['c']
	
	# library
	from scipy.integrate import odeint
		
	# Lorentz function
	def L96(xy, t):
		"""Lorenz 96 fast-slow model with constant forcing"""
		x = xy[:I]
		y = xy[I:].reshape(J, I)
		
		dxdt = (x.take(range(1, I + 1), mode='wrap') - x.take(range(-2, I - 2), mode='wrap')) * x.take(range(-1, I - 1), mode='wrap') - x + F - h * c / b * y.sum(axis=0)
		dydt = c * b * y.take(range(1, J + 1), mode='wrap', axis=0) * (y.take(range(-1, J - 1), mode='wrap', axis=0) - y.take(range(1, J + 1), mode='wrap', axis=0)) - c * y + h * c / b * _np.tile(x, (J, 1))
			
		d_dt = _np.concatenate((dxdt, dydt.reshape(-1)))
		return d_dt
	
	# setup time
	t = _np.arange(0.0, float(t_final), dt)
	
	# setup ICs
	x0 = F * _np.ones(I)  # Initial state (equilibrium)
	y0 = F * _np.ones((J, I))  # Initial state (equilibrium)
	x0[0] += 0.01  # Add small perturbation to the first variable
	
	# solve
	out = odeint(func=L96, y0=_np.concatenate((x0, y0.reshape(-1))), t=t)
	
	# format as xarray
	x_out = out[:, :I]
	y_out = out[:, I:].reshape((len(t), J, I))
	i = _np.arange(0, I)
	j = _np.arange(0, J)
	x_out = _xr.DataArray(x_out, dims=['t', 'i'], coords=[t, i])
	y_out = _xr.DataArray(y_out, dims=['t', 'j', 'i'], coords=[t, j, i])
	
	# only return the second half of the data
	x_out = x_out[(len(t)//2):, :]
	y_out = y_out[(len(t)//2):, :, :]
	
	# TEMPORARY.  Plot FFT of each signal
	if True:
		from johnspythonlibrary2.Process.Spectral import fft
		
		x_sel = x_out.sel(i=I//2)
		x_sel -= x_sel.mean()
		fft(x_sel, plot=True, trimNegFreqs=True)
		
		y_sel = y_out.sel(i=I//2, j=J//2)
		y_sel -= y_sel.mean()
		fft(y_sel, plot=True, trimNegFreqs=True)
	
	return x_out, y_out


def lorentz96(N=5, F=8, t_final=30.0, dt=0.01, plot=False):
	"""
	Parameters
	----------
	N : int
		number of variables
	F : float
		Forcing value
	t_final : float
		Duration of simulation
	dt : float
		Time step
	plot : bool
		Optional plot
	
	References
	----------
	* https://en.wikipedia.org/wiki/Lorenz_96_model
	* 'a common benchmark for spatiotemporal filtering methods due to its chaotic dynamics and moderate dimensionality' : https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.011021
	
	Examples
	--------
	
	Example 1::
		
		x = lorentz96(plot=True)
		
	Example 2::
		
		x = lorentz96(40, t_final=50, plot=True)
	"""
	
	if N < 4:
		raise Exception("N must be >= 4")
	
	# parameters

	# library
	from scipy.integrate import odeint
		
	# subfunction
	def L96(x, t):
		"""Lorenz 96 model with constant forcing"""
		d = (x.take(range(1,N+1), mode='wrap') - x.take(range(-2,N-2), mode='wrap')) * x.take(range(-1, N-1), mode='wrap') - x + F
		return d
	
	# setup time
	t = _np.arange(0.0, float(t_final), dt)
	
	# setup ICs
	x0 = F * _np.ones(N)  # Initial state (equilibrium)
	x0[0] += 0.01  # Add small perturbation to the first variable
	
	# solve
	x = odeint(L96, x0, t)
	
	# finalize
	x = _xr.DataArray(x.transpose(), dims=['n', 't'], coords=[_np.arange(0, N) + 1, t])
	x.t.attrs = {'long_name': 'Time', 'units': 'au'}
	
	if plot is True:
		fig = _plt.figure()
		ax = fig.gca(projection="3d")
		ax.plot(x.sel(n=1), x.sel(n=2), x.sel(n=3))
		ax.set_xlabel("$x_1$")
		ax.set_ylabel("$x_2$")
		ax.set_zlabel("$x_3$")
		
		fig, ax = _plt.subplots()
		x.plot(ax=ax)
		
	return x


def van_der_pol_oscillator():
    print("TODO")
    # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    

def chen_and_lu_systems():
    print("TODO")
    # https://www.sciencedirect.com/science/article/pii/S0096300314017937
    

def rossler_attractor():
    print("TODO")
    # https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    
    
def plot_double_pendulum_single_frame(x, i, filename_root=None):
    print("work in progress")
    
    from matplotlib.patches import Circle
    
    r = 0.05 # bob circle radius
    
    x1 = float(x.x1[i])
    x2 = float(x.x2[i])
    y1 = float(x.y1[i])
    y2 = float(x.y2[i])
    L1 = x.attrs["system_params"]["L1"]
    L2 = x.attrs["system_params"]["L2"]
    
    fig, ax = _plt.subplots()
    ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1, y1), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2, y2), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # # The trail will be divided into ns segments and plotted as a fading line.
    # ns = 20
    # s = max_trail // ns

    # for j in range(ns):
    #     imin = i - (ns-j)*s
    #     if imin < 0:
    #         continue
    #     imax = imin + s + 1
    #     # The fading looks better if we square the fractional length along the
    #     # trail.
    #     alpha = (j/ns)**2
    #     ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
    #             lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if type(filename_root) is not type(None):
        fig.savefig(f"{filename_root}_{i:06d}.png")
        _plt.close(fig)
        
    return fig, ax
        
        
def animate_double_pendulum(x, filename_root="temp", fps=30, filepath_dir="."):
    """ 
    
    Examples
    --------
    
    Example 1::
        
        results = double_pendulum()
        animate_double_pendulum(results)
    """
    _plt.ioff()
    
    N = len(x.t)
    
    for i in range(N):
        plot_double_pendulum_single_frame(x, i, filename_root=filename_root)
    
    from johnspythonlibrary2.VideoImage import create_gif_from_sequential_images
    from pathlib import Path
    gif_filename = filename_root + ".gif"
    image_filepaths = Path(filepath_dir).glob(f"{filename_root}*.png")
    
    create_gif_from_sequential_images(image_filepaths, gif_filename, fps)
    
    _plt.ion()
    
    
    
def plot_double_pendulum(results):
    

    # jpl2.Process.Spectral.wrapPhase(results.theta2).plot()
    fig, axes = _plt.subplots(3, sharex=True, )
    
    # torque = extract_torque_from_torque_func(t, torque_func, plot=False)
    
    # torque.torque1.plot(ax=axes[0], label="torque1")
    # torque.torque2.plot(ax=axes[0], label="torque2")
    (results.theta1 / (_np.pi * 2)).plot(ax=axes[0], label="theta1")
    (results.theta2 / (_np.pi * 2)).plot(ax=axes[0], label="theta2")
    (results.dtheta1dt / (_np.pi * 2)).plot(ax=axes[1], label="theta1")
    (results.dtheta2dt / (_np.pi * 2)).plot(ax=axes[1], label="theta2")
    results.energy.plot(ax=axes[2], label="total energy")
    results.energy_kinetic.plot(ax=axes[2], label="kinetic energy")
    results.energy_potential.plot(ax=axes[2], label="potential energy")
    ylabels = ["Angular position\n(periods)", "Angular velocity\n(periods/s)", "Energy"]
    for i, ax in enumerate(axes):
        ax.legend()
        ax.axhline(0, ls="--", color="grey")
        ax.set_ylabel(ylabels[i])
        
    
def double_pendulum(
        time_params=dict(tmax=30, dt=0.01, t0=0,),
        system_params=dict(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81, drag_coef1=0.0, drag_coef2=0.0,), # Pendulum rod lengths (m), bob masses (kg), The gravitational acceleration (m.s-2)
        ICs=dict(theta1=3 * _np.pi / 7, dtheta1dt=0, theta2 = 3 * _np.pi / 4, dtheta2dt=0), # theta1, d(theta1)/dt, theta2, d(theta2)/dt
        torque_func=None,
        # torque_func = lambda x, y: (1., 1.)
        ):
    """
    
    Examples
    --------
    
    Example 1 :: 
        ## test conservation of energy with a small perturbation and no external torque
        results = double_pendulum(
                time_params=dict(tmax=30, dt=0.001, t0=0, time_lags_in_dt=[1]),
                system_params=dict(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81, drag_coef1=0.0, drag_coef2=0.0), # Pendulum rod lengths (m), bob masses (kg), The gravitational acceleration (m.s-2)
                ICs=dict(theta1=(2 + 0.1) * np.pi, dtheta1dt=0, theta2 = 2 * np.pi, dtheta2dt=0), # theta1, d(theta1)/dt, theta2, d(theta2)/dt
                )
        
        plot_double_pendulum_single_frame(x=results, i=0)
        
        plot_double_pendulum(results)
    
    Example 2::
        
        results = double_pendulum(
            torque_func = lambda x, y: (1.25, 1.25),
            )
        
        import matplotlib.pyplot as plt
        plt.plot(results.x2, results.y2)
        
    Example 3::
        
        results = double_pendulum(
            system_params=dict(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81, drag_coef1=0, drag_coef2=0, ),
            time_params=dict(tmax=t_final / 10.0, dt=dt / 2.0, t0=t_initial, time_lags_in_dt=[1]),
            ICs=dict(theta1=0.25 * np.pi / 2, dtheta1dt=0, theta2 = 0 * np.pi / 2, dtheta2dt=0),
            torque_func=None,
            )
        
        plot_double_pendulum(results)
        
        plot_double_pendulum_single_frame(results, i=0)
    
    References
    ----------
     * https://scipython.com/blog/the-double-pendulum/
     * https://medium.com/@hamxa26/a-dance-of-chaos-simulating-a-double-pendulum-dcbf4f96dd16
    """
    
    import numpy as np
    from scipy.integrate import odeint
    
    ## system parameters
    L1, L2, m1, m2, g, drag_coef1, drag_coef2 = map(system_params.get, ("L1", "L2", "m1", "m2", "g", "drag_coef1", "drag_coef2"))
    
    ## initial conditions
    theta1, dtheta1dt, theta2, dtheta2dt = map(ICs.get, ("theta1", "dtheta1dt", "theta2", "dtheta2dt"))
    x0 = np.array([theta1, dtheta1dt, theta2, dtheta2dt])
    
    ## time domain
    tmax, dt, t0 = map(time_params.get, ("tmax", "dt", "t0"))
    t = np.arange(t0, tmax+dt, dt)
    
    ## right hand side of the ODEs
    def RHS(x_i, t, L1, L2, m1, m2, torque_func=None):
        theta1, z1, theta2, z2 = x_i

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
        
        ## applied torque
        if type(torque_func) is type(None):
            T1_div_I1 = 0
            T2_div_I2 = 0
        else:
            # print("work in progress")
            torques = torque_func(x_i, t)
            T1, T2 = torques[2:4]
            T1_div_I1 = T1 / (L1 * m1)
            T2_div_I2 = T2 / (L2 * m2)
            
        ## viscous damping
        drag1 = drag_coef1 * z1 / (L1 * m1)
        drag2 = drag_coef2 * z2 / (L2 * m2)
        
        ## RHS equations
        theta1dot = z1
        z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
                 (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2) + T1_div_I1 - drag1
        theta2dot = z2
        z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
                 m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2) + T2_div_I2 - drag2
        
        return theta1dot, z1dot, theta2dot, z2dot
    
    # Do the numerical integration of the equations of motion
    x = odeint(RHS, x0, t, args=(L1, L2, m1, m2, torque_func))

    # Unpack z and theta as a function of time
    # theta1, theta2 = x[:,0], x[:,2]
    theta1, dtheta1dt, theta2, dtheta2dt = x[:,0], x[:, 1], x[:,2], x[:, 3]
    
    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    # energy - https://www.jousefmurad.com/engineering/double-pendulum-1/#:~:text=The%20kinetic%20energy%20T%20can%20be%20expressed%20as%3A,are%20the%20velocities%20of%20the%20two%20pendulum%20bobs.
    energy_kinetic = 0.5  * m1 * (L1 * dtheta1dt)**2 + 0.5 * m2 * ((L1 * dtheta1dt)**2 + (L2 * dtheta2dt)**2 +
            2 * L1 * L2 * dtheta1dt * dtheta2dt * np.cos(theta1-theta2))
    energy_potential = -m1 * g * L1 * (np.cos(theta1) - 1) - m2 * g * (L1 * (np.cos(theta1) - 1) + L2 * (np.cos(theta2) - 1))
    energy = energy_kinetic + energy_potential
    
    # convert to xarray
    t = _xr.DataArray(t, coords={"t": t})
    x1 = _xr.DataArray(x1, coords=[t])
    x2 = _xr.DataArray(x2, coords=[t])
    y1 = _xr.DataArray(y1, coords=[t])
    y2 = _xr.DataArray(y2, coords=[t])
    theta1 = _xr.DataArray(theta1, coords=[t])
    theta2 = _xr.DataArray(theta2, coords=[t])
    dtheta1dt = _xr.DataArray(dtheta1dt, coords=[t])
    dtheta2dt = _xr.DataArray(dtheta2dt, coords=[t])
    energy_kinetic = _xr.DataArray(energy_kinetic, coords=[t])
    energy_potential = _xr.DataArray(energy_potential, coords=[t])
    energy = _xr.DataArray(energy, coords=[t])
    results = _xr.Dataset(dict(x1=x1, x2=x2, y1=y1, y2=y2, theta1=theta1, theta2=theta2, dtheta1dt=dtheta1dt, dtheta2dt=dtheta2dt, energy=energy, energy_kinetic=energy_kinetic, energy_potential=energy_potential), 
                          attrs=dict(time_params=time_params, system_params=system_params, ICs=ICs, ))
    
    return results


def lorentzAttractor(	N=2000,
						dt=0.05,
						ICs={	'x0': -9.38131377,
								'y0': -8.42655716 , 
								'z0': 29.30738524},
						plot=False,
						removeMean=False,
						normalize=False,
						removeFirstNPoints=500,
						args={	'sigma': 10.0,
								'b': 8.0 / 3.0,
								'r': 28.0}):
	"""
	Solves the lorentz attractor nonlinear ODEs.
	
	References
	----------
	 * https://en.wikipedia.org/wiki/Lorenz_system
	 * https://www.diasporist.org/jeroen/assets/lecture%20-%20A%20brief%20introduction%20to%20the%20Lorenz%2063%20system.pdf
	 
	Examples
	--------
	Example 1::
		
		_plt.close('all')
		x, y, z = lorentzAttractor( N=5000,
									dt = 0.02,
									ICs={	'x0': -9.38131377,
											'y0': -8.42655716 , 
											'z0': 29.30738524},
									plot = 'all',
									removeMean = True,
									normalize = True,
									removeFirstNPoints = 500)
						
	"""
	
	from scipy.integrate import solve_ivp
	
	## initialize
	N += removeFirstNPoints
	T = N * dt
	ICs = _np.array(list(ICs.items()))[:, 1].astype(float)
	
	## Solve Lorentz system of equations
	def ODEs(t, y, *args):
		X, Y, Z = y
		sigma, b, r = args
		derivs =	[	sigma * (Y - X),
						-X * Z + r * X - Y,
						X * Y - b * Z]
		return derivs
	
	t_eval = _np.arange(0, T, dt)
	psoln = solve_ivp(	ODEs,
						[0, T],
						ICs,  # initial conditions
						args=args.values(),
						t_eval=t_eval
						)
	x, y, z = psoln.y
	
	## Cleanup data
	x = _xr.DataArray(	x,
						dims=['t'],
						coords={'t': t_eval},
						attrs={	'units': "au",
								'long_name': 'Amplitude'})
	x.t.attrs = {'units': 'au', 'long_name': 'Time'}
	y = _xr.DataArray(	y,
						dims=['t'],
						coords={'t': t_eval},
						attrs={	'units': "au",
								'long_name': 'Amplitude'})
	y.t.attrs = {'units': 'au', 'long_name': 'Time'}
	z=_xr.DataArray(	z,
						dims=['t'],
						coords={'t': t_eval},
						attrs={	'units': "au",
								'long_name': 'Amplitude'})
	z.t.attrs = {'units': 'au', 'long_name': 'Time'}
	
	if removeFirstNPoints > 0:
		x = x[removeFirstNPoints:]
		y = y[removeFirstNPoints:]
		z = z[removeFirstNPoints:]
	
	if removeMean is True:
		x -= x.mean()
		y -= y.mean()
		z -= z.mean()
		
	if normalize is True:
		x /= x.std()
		y /= y.std()
		z /= z.std()
		
	ds=_xr.Dataset({	'x':x,
						'y':y,
						'z':z})
			
	## optional plots
	if plot is not False:
		fig, ax = _plt.subplots(3, sharex=True)
		markersize = 2
		x.plot(ax=ax[0], marker='.', markersize=markersize)
		y.plot(ax=ax[1], marker='.', markersize=markersize)
		z.plot(ax=ax[2], marker='.', markersize=markersize)
		ax[0].set_title('Lorentz Attractor\n' + r'($\sigma$, b, r)=' + '(%.3f, %.3f, %.3f)' % (args['sigma'], args['b'], args['r']) + '\nICs = (%.3f, %.3f, %.3f)' % (ICs[0], ICs[1], ICs[2]))
		
	if plot == 'all':
		_plt.figure(); _plt.plot(x, y)
		_plt.figure(); _plt.plot(y, z)
		_plt.figure(); _plt.plot(z, x)
	
		import matplotlib as _mpl
		_mpl.rcParams.update({'figure.autolayout': False})
		fig = _plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax.plot(x, y, zs=z)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')		
		ax.set_title('Lorentz Attractor\n' + r'($\sigma$, b, r)=' + '(%.3f, %.3f, %.3f)' % (args['sigma'], args['b'], args['r']) + '\nICs = (%.3f, %.3f, %.3f)' % (ICs[0], ICs[1], ICs[2]) + '\nState space')
		
		ax = fig.add_subplot(122, projection='3d')
		ax.plot(x[6::3], x[3:-3:3], zs=x[:-6:3])
		ax.set_xlabel('x(t)')
		ax.set_ylabel(r'x(t-$\tau$)')
		ax.set_zlabel(r'x(t-2$\tau$)')
		ax.set_title('Lorentz Attractor\n' + r'($\sigma$, b, r)=' + '(%.3f, %.3f, %.3f)' % (args['sigma'], args['b'], args['r']) + '\nICs = (%.3f, %.3f, %.3f)' % (ICs[0], ICs[1], ICs[2]) + '\nTime-lagged state space')
		fig.tight_layout(pad=5, w_pad=5)
		
	return ds


def predatorPrey(	N=100000,
					T=100,
					ICs={	'x0': 0.9,
							'y0': 0.9},
					args={	'alpha': 2.0 / 3.0,
							'beta': 4.0 / 3.0,
							'delta': 1.0,
							'gamma': 1.0},
					plot=False,
					removeMean=False,
					normalize=False):
	"""
	Lotka–Volterra (classic predator-prey problem) equations
	
	References
	----------
	* https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
	
	Examples
	--------
	Example 1::
		
		predatorPrey( 	N=100000,
						T=100,
						ICs={	'x0': 0.9,
								'y0': 0.9},
						args={	'alpha': 2.0 / 3.0,
								'beta': 4.0 / 3.0,
								'delta': 1.0,
								'gamma': 1.0},
						plot = True)
		
	Example 2::
		
		predatorPrey( 	N=100000,
						T=100,
						ICs={	'x0': 10,
								'y0': 15},
						args={	'alpha': 1.1,
								'beta': 0.4,
								'delta': 0.1,
								'gamma': 0.4},
						plot = True)
	"""
	
	from scipy.integrate import solve_ivp
	
	## initialize
	dt=T/N
	
	## Solve Lorentz system of equations
	def ODEs(t, y, *args):
		X,Y = y
		alpha, beta, delta, gamma = args
		#print(X,Y,alpha,beta,delta,gamma)
		derivs=	[	alpha * X - beta * X * Y,
					delta * X * Y - gamma * Y]
		return derivs
	
	t_eval = _np.arange(0, T, dt)
	psoln = solve_ivp(	ODEs,
						[0, T],
						_np.array(list(ICs.items()))[:, 1].astype(float),  # initial conditions
						args=args.values(),
						t_eval=t_eval
						)
	x, y = psoln.y
	
	## Cleanup data
	x = _xr.DataArray(	x,
						dims=['time'],
						coords={'time': t_eval},
						attrs={'units': "au",
								 'standard_name': 'prey'})
	x.time.attrs = {'units': 'au'}
	y = _xr.DataArray(	y,
						dims=['time'],
						coords={'time': t_eval},
						attrs={'units': "au",
								 'standard_name': 'predator'})
	y.time.attrs = {'units': 'au'}
	
	if removeMean is True:
		x -= x.mean()
		y -= y.mean()
		
	if normalize is True:
		x /= x.std()
		y /= y.std()
		
	ds = _xr.Dataset({	'x': x,
						'y': y})
	
	## Optional plots
	if plot is True:
		fig, ax = _plt.subplots(2, sharex=True)
		ds.x.plot(ax=ax[0])
		ds.y.plot(ax=ax[1])
		
		fig, ax = _plt.subplots()
		ax.plot(ds.x.values, ds.y.values)
		ax.set_xlabel('x, prey')
		ax.set_ylabel('y, predator')
	
	return ds
	
	
def coupledHarmonicOscillator(	N=100000,
								T=10000,
								ICs={	'y1_0': 0,
										'x1_0': 1,
										'y2_0': 0,
										'x2_0': 0},
								args={	'k': 4,
										'kappa': 2,
										'm': 1},
								plot=False):
	"""
	Solve a coupled harmonic oscillator problem. See reference for details.
	
	Examples
	--------
	Example1 ::
		ds = coupledHarmonicOscillator(plot=True, N=1e4, T=1)
	
	References
	----------
	http://users.physics.harvard.edu/~schwartz/15cFiles/Lecture3-Coupled-Oscillators.pdf

	"""
	
	import matplotlib.pyplot as _plt
	
	dt  = T / N
	
	from scipy.integrate import solve_ivp
	 
	def ODEs(t, y, *args):
		y1, x1, y2, x2 = y
		k, kappa, m = args
		derivs = [	-(k + kappa) / m * x1 + kappa / m * x2,
					y1,  
 					-(k + kappa) / m * x2 + kappa / m * x1,
					y2]
		return derivs

	time = _np.arange(0, T, dt)
	psoln = solve_ivp(	ODEs,
						t_span=[0, T],
						y0=_np.array(list(ICs.values())),  # initial conditions
						args=args.values(),
						t_eval=time
					)
	
	y1, x1, y2, x2 = psoln.y
	
	x1=_xr.DataArray(	x1,
						dims=['time'],
						coords={'time': time},
						attrs={'units': "au",
								 'standard_name': 'Amplitude'})
	x1.time.attrs = {'units': 'au'}
	x2=_xr.DataArray(	x2,
						dims=['time'],
						coords={'time': time},
						attrs={'units': "au",
								 'standard_name': 'Amplitude'})
	x2.time.attrs={'units': 's'}
    
	## characteristic frequencies
	f_fast = _np.sqrt((args["k"] + 2 * args["kappa"]) / args["m"]) / (2 * _np.pi)
	f_slow = _np.sqrt((args["k"] + 0) / args["m"]) / (2 * _np.pi)
	
	if plot==True:
		fig, ax = _plt.subplots(2, sharex=True)
		x1.plot(ax=ax[0], marker='.')
		x2.plot(ax=ax[1], marker='.')
		ax[0].set_title('Coupled harmonic oscillator\n' + r'(k, kappa, m)='+'(%.3f, %.3f, %.6f)' % (args['k'], args['kappa'], args['m']) + '\nICs = (%.3f, %.3f, %.3f, %.3f)' % (ICs['y1_0'], ICs['x1_0'], ICs['y2_0'], ICs['x2_0']))

		fig, ax = _plt.subplots()
		ax.plot(x1.values, x2.values)
		ax.set_xlabel('x1')
		ax.set_xlabel('x2')

	ds=_xr.Dataset(	{'x1': x1,
					 'x2': x2},
                    attrs = {'f_fast': f_fast, 'f_slow': f_slow})
	
	return ds


def coupledHarmonicOscillator_nonlinear(	N=500,
											dt=0.1,
											ICs={	'y1_0': 0,
													'x1_0': 1,
													'y2_0': 0,
													'x2_0': 0},
											args={	'f1': 45,
													'f2': 150,
													'm': 1,
													'E': 1e5},
											plot=False,
											verbose=False):
	"""
	Solve a coupled harmonic oscillator problem. See reference for details.
	
	Examples
	--------
	Example1 ::
		ds = coupledHarmonicOscillator_nonlinear(N=10000, dt=1e-3, plot=True, verbose=False)
		nperseg=2000
		from johnspythonlibrary2.Process.Spectral import fft_average
		fft_average(ds.x1,plot=True,nperseg=nperseg,noverlap=nperseg//2)
	
	References
	----------
	https://arxiv.org/pdf/1811.02973.pdf

	"""
	
	f1, f2, m, E = args.values()
	k1 = (2 * _np.pi * f1)**2 * m
	k2 = ((2 * _np.pi * f2)**2 * m - k1) / 2
	
	import matplotlib.pyplot as _plt
	
	from scipy.integrate import solve_ivp
	 
	def ODEs(t, y, *args):
		y1, x1, y2, x2 = y
		k1, k2, m1, m2, E = args
		if verbose:
			print(-k1 * x1 / m1, +k2 * (x2 - x1) / m1, E * x1**2 / m1)
		derivs=	[	-k1 * x1 / m1 + k2 * (x2 - x1) / m1 + E * x1**2 / m1,
					y1,  
 					-k1 * x2 / m2 - k2 * (x2 - x1) / m2,
					y2]
		return derivs

	time = _np.arange(0, N) * dt
	T = time[-1]
	psoln = solve_ivp(	ODEs,
						t_span=[0, T],
						y0=_np.array(list(ICs.values())),  # initial conditions
						args=(k1, k2, m, m, E),
						t_eval=time
					)
	
	y1, x1, y2, x2 = psoln.y
	
	x1=_xr.DataArray(	x1,
						dims=['t'],
						coords={'t': time},
						attrs={'units': "au",
								 'standard_name': 'Amplitude'})
	x1.t.attrs = {'units': 'au'}
	x2 = _xr.DataArray(	x2,
						dims=['t'],
						coords={'t': time},
						attrs={'units': "au",
								 'standard_name': 'Amplitude'})
	x2.t.attrs = {'units': 's'}
	
	if plot is True:
		fig, ax = _plt.subplots(2, sharex=True)
		x1.plot(ax=ax[0], marker='.')
		x2.plot(ax=ax[1], marker='.')
		ax[0].set_title('Coupled harmonic oscillator\n' + r'(f1, f2, m, E)=' + '(%.3f, %.3f, %.3f,%.3f)' % (f1, f2, m, E) + '\nICs = (%.3f, %.3f, %.3f, %.3f)' % (ICs['y1_0'], ICs['x1_0'], ICs['y2_0'], ICs['x2_0']))
		
		fig, ax = _plt.subplots()
		ax.plot(x1.values, x2.values)
		ax.set_xlabel('x1')
		ax.set_xlabel('x2')
		
	ds=_xr.Dataset(	{'x1': x1,
					 'x2': x2})
	
	return ds
