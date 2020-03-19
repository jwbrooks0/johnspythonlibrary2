
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


