
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt


def polyFitData(df,	
				order=2, 
				plot=True,
				verbose=True):
	""" 
	Polynomial fit function.  
	
	Parameters
	----------
	df : pandas.core.frame.DataFrame
		Dataframe with a single column
		index = independent variable
	order : int
		order of polynomial fit.  1 = linear, 2 = quadratic, etc.
	plot : bool
		Causes a plot of the fit to be generated
	verbose : bool
		print additional details
	
	Returns
	-------
	dfFit : pandas.core.frame.DataFrame with a the fit
		Dataframe with a single column
		index = independent variable
	ffit : numpy.poly1d
		fit function
	
	Example
	-------
	::
		
		 x=np.arange(0,1,0.1)
		 y=2*x+1
		 df=_pd.DataFrame(y,index=x)
		 polyFitData(df,order=1,plot=True)
		 
	"""
	
	coefs=_np.polyfit(	df.index.to_numpy(), 
						df.iloc[:,0].to_numpy(), 
						deg=order)
	ffit = _np.poly1d(	coefs)
	dfFit=_pd.DataFrame(	ffit(df.index.to_numpy()),
								index=df.index.to_numpy())
	if verbose:
		print("fit coeficients:")
		print(coefs)
		
	if plot==True:
		x=_np.linspace(dfFit.index[0],dfFit.index[-1],1000)
		y=ffit(x)
		
		fig,ax=_plt.subplots()
		ax.plot(df,'x',linestyle='',label='data')
		ax.plot(x,y,label='fit')
		ax.legend()
		
	return dfFit,ffit


def _rSquared(y,f):
	"""
	calculates R^2 of data fit
		
	Parameters
	----------
	y : numpy.ndarray
		data being fit to, the dependent variable (NOT THE INDEPENDENT VARIABLE).  y is a functino of x, i.e. y=y(x)
	f : float
		fit data
		
	Returns
	-------
	: float 
		R^2 = 1 - \frac{\sum (f-y)^2 }{\sum (y-<y>)^2 }
	
	Reference
	---------
	https://en.wikipedia.org/wiki/Coefficient_of_determination
	"""
	yAve=_np.average(y);
	SSres = _np.sum( (y-f)**2 )
	SStot = _np.sum( (y-yAve)**2 )
	return 1-SSres/SStot

	
class _genericLeastSquaresFit:
	"""
	Least squares fitting function(class)
	This is a wrapper for scipy.optimize.least_squares
	
	Parameters
	----------
	y : numpy.ndarray
		dependent data
	x : numpy.ndarray (multidimensional)
		independent array(s).  multiple ind. variables are supported.  
	paramsGuess : list (of floats)
		guess values for the fit parameters
	function : function
		function to attempt to fit the data to.  see exdamples _cosFunction and 
		_expFunction to see how this function should be constructed
	yTrue : (optional) numpy.ndarray 
		if you know what the actual fit should be (for example when testing 
		this code against a known), include it here, and it will plot 
		alongside the other data.  also useful for debugging.  
	plot : bool
		causes the results to be plotted
		
	Attributes
	----------
	fitParams : numpy.ndarray
		array of fit parameters, in the same order as the guess
	plotOfFit :
	plotOfFitDep :
		custom plot function of fit data plotted against dependent data.  this 
		is important if there is more than 1 dependent data array.
	rSquared : float
		r^2 result of the fit
	res : 
		fit output from scipy.optimize.least_squares
	yFit : numpy.ndarray
		y-fit data that corresponds with the independent data
	
	Notes
	-----
	i've found that the guess values often NEED to be somewhat close to the 
	actual values for the solution to converge
	
	i've implemented several specific functions that implement this function.
	see expFit and cosFit
	
	Example use
	------------
	::
		
		# define expoential function
		def _expFunction(x, a,b,c):
			return a*_np.exp(x/b)+c
		# generate noisy exponential signal
		x1=_np.linspace(-1,1,100);
		a=_np.zeros(len(x1));
		b=_np.zeros(len(x1));
		c=_np.zeros(len(x1));
		for i in range(0,len(x1)):
			a[i]=(random.random()-0.5)/4. + 1.
			b[i]=(random.random()-0.5)/4. + 1.
			c[i]=(random.random()-0.5)/4. + 1.
		y1=1+_np.pi*_np.exp(x1/_np.sqrt(2)) # actual solution
		y2=1*a+_np.pi*b*_np.exp(x1/_np.sqrt(2)/c) # noisy solution
		# perform fit
		d=genericLeastSquaresFit(x1,[1,1,1],y2, _expFunction, y1,plot=True)
	"""		   
	
	def __init__(self, x, paramsGuess, y, function,yTrue=[],plot=True ):

		def fit_fun(paramsGuess,x,y):
			return function(x, *paramsGuess) - y

		from scipy.optimize import least_squares
		
		self.x=x
		self.y=y
		self.yTrue=yTrue
		
		# perform least squares fit and record results
		self.res=least_squares(fit_fun, paramsGuess, args=[x,y])  #args=(y)
		self.yFit=function(x, *self.res.x) 
		self.fitParams=self.res.x;
		
		# calculate r^2
		self.rSquared=_rSquared(y,self.yFit)
		
		# print results to screen
		print(r'R2 =  %.5E' % self.rSquared)
		print('fit parameters')
		print('\n'.join('{}: {}'.format(*k) for k in enumerate(self.res.x)))
		
		# plot data
		if plot==True:
			if type(x) is _np.ndarray or len(x)==1:
				self.plotOfFit().show()
#			self.plotOfFitDep().plot()
		
  
	
	def plotOfFit(self):
		"""
		plots raw and fit data vs. its indep. variable.  
		
		Notes
		-----
		this function only works if there is a single indep. variable
		"""  
		  
		# make sure that there is only a single indep. variable
		if type(self.x) is _np.ndarray or len(self.x)==1:
			p1,ax1=_plt.subplots()
			ax1.set_ylabel('y')
			ax1.set_xlabel('x')
			ax1.set_title=(r'Fit results.  R$^2$ = %.5f' % self.rSquared)
			
			if isinstance(self.x,list):
				x=self.x[0]
			else:
				x=self.x
				
			# raw data
			ax1.plot(x,self.y,'b.',alpha=0.3,label='raw data')
			
			# fit data
			ax1.plot(x,self.yFit,'b',alpha=1,label='fit')

			return p1
		


def _cosFunction(x, a,b,c,d):
	"""
	Cosine function.  Used primarily with fitting functions.  
	Output = a*_np.cos(x*d*2*_np.pi+b)+c
	
	Parameters
	----------
	x : numpy.ndarray
		Independent variable
	a : float
		Fitting parameter.  Output = a*_np.cos(x*d*2*_np.pi+b)+c
	b : float
		Fitting parameter.  Output = a*_np.cos(x*d*2*_np.pi+b)+c
	c : float
		Fitting parameter.  Output = a*_np.cos(x*d*2*_np.pi+b)+c
	d : float
		Fitting parameter.  Output = a*_np.cos(x*d*2*_np.pi+b)+c
		
	Returns
	-------
	: numpy.ndarray
		Output = a*_np.cos(x*d*2*_np.pi+b)+c
	"""
	return a*_np.cos(x*d*2*_np.pi+b)+c
	

	
class cosTimeFit:
	"""
	Cos fit function.  a*_np.cos(x*d*2*_np.pi+b)+c

	Parameters
	----------
	y : numpy.ndarray
		dependent data
	x : numpy.ndarray
		independent array
	guess : list
		list of four floats [a, b, c, d]=[amplitude, phase offset, amplitude offest, linear frequency].  these are the guess values.
	plot : bool
		causes the results to be plotted
		
	Attributes
	----------
	fit : genericLeastSquaresFit
	
	Notes
	-----
	if you are receiving the error: "ValueError: Residuals are not finite in 
	the initial point.", most likely, you need to play with your initial 
	conditions to get them closer to the right answer before the fit will work
	correctly.  
	
	Example use
	-----------
	::
		
		import numpy as np
		
		y=np.array([11.622967, 12.006081, 11.760928, 12.246830, 12.052126, 12.346154, 12.039262, 12.362163, 12.009269, 11.260743, 10.950483, 10.522091,  9.346292,  7.014578,  6.981853,  7.197708,  7.035624,  6.785289, 7.134426,  8.338514,  8.723832, 10.276473, 10.602792, 11.031908, 11.364901, 11.687638, 11.947783, 12.228909, 11.918379, 12.343574, 12.046851, 12.316508, 12.147746, 12.136446, 11.744371,  8.317413, 8.790837, 10.139807,  7.019035,  7.541484,  7.199672,  9.090377,  7.532161,  8.156842,  9.329572, 9.991522, 10.036448, 10.797905])
		x=np.linspace(0,2*np.pi,48)
		c=cosTimeFit(y,x,guess=[2,0,10,.3])
		# note that this example took me quite a bit of guessing with the guess 
		# values before everything fit correctly.

	"""
	def __init__(self,y,x,guess,plot=True):
		self.fit=_genericLeastSquaresFit(x=x,paramsGuess=guess,y=y, 
										function=_cosFunction,plot=plot)


