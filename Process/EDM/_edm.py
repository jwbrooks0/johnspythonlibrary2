
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import time as _time
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Plot import finalizeFigure as _finalizeFigure
from johnspythonlibrary2.Plot import legendOutside as _legendOutside


###################################################################################
### sub-functions
def printTime(string,start):
	""" print time since start """
	if string!='':
		print('%s'%(string))
	print('%.3fs'%((_time.time()-start)))


def createSignal(N=1000,plot=False):
	""" generate fake data using a tentmap """
	t=_np.arange(0,N+1)
	x=_np.zeros(t.shape,dtype=_np.float128)
	x[0]=_np.sqrt(2)/2.0
	def tentMap(x,mu=_np.float128(2-1e-15)):
		for i in range(1,x.shape[0]):
			if x[i-1]<0.5:
				x[i]=mu*x[i-1]
			elif x[i-1]>=0.5:
				x[i]=mu*(1-x[i-1])
			else:
				raise Exception('error')
		return x
	sx=_pd.Series(tentMap(x),index=t)
	
	# the actual data of interest is the first difference of x
	sxDelta=sx.iloc[1:].values-sx.iloc[0:-1]
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(sx,linestyle='-',marker='.',linewidth=0.5,markersize=3)
		ax[1].plot(sxDelta,linestyle='-',marker='.',linewidth=0.5,markersize=3)
		
	return sxDelta



def splitData(s):
	""" split data into testing (sY) and training (sX) data sets """
	if _np.mod(s.shape[0],2)==1:
		s=s.iloc[:-1]
	s.index=_np.arange(0,s.shape[0])
	sX=s.iloc[0:s.shape[0]//2]
	sY=s.iloc[s.shape[0]//2:]
	return sX,sY,s

	
def convertToStateSpace(s,E,tau):
	""" Convertst input to parameter space using the embedded dimension, E """
	index=s.index.values[(E-1)*tau:]
	columns=_np.arange(-(E-1)*tau,1,tau)
	dfs=_pd.DataFrame(index=index,columns=columns)	
	for key,val in dfs.iteritems():
		if key==0:
			dfs[key]=s.iloc[key-columns.min():s.shape[0]+key].values	
		else:
			dfs[key]=s.iloc[key-columns.min():s.shape[0]+key].values	
	return dfs


def weighting(radii,method='exponential'):
	""" Weights to be applied nearest neighbors """
	
	if method =='exponential':
		weights=_np.exp(-radii/radii.min(axis=1)[:,_np.newaxis])
		weights/=weights.sum(axis=1)[:,_np.newaxis] 	 
	elif method =='uniform':
		weights=_np.ones(radii.shape)/radii.shape[1]
	
	weights=_pd.DataFrame(weights,index=radii.index)
	return weights



def forecasting(sx,dfY,keysOfNearestNeighbors,weights,T):
	""" The forecasting method.  Combines weights with correct indices (keys) to get the forecast """
	dfTActual=_pd.DataFrame(index=dfY.index.values[:-T+1],dtype=float)
	for key in range(1,1+T):
		dfTActual['%d'%key]=dfY.iloc[key-1:dfY.shape[0]-T+key][0].values	
	dfTGuess=_pd.DataFrame(index=dfTActual.index,columns=dfTActual.columns,dtype=float)
	
	for key,val in dfTGuess.iteritems():
			shape=keysOfNearestNeighbors.shape
			y=_pd.DataFrame(sx.loc[(keysOfNearestNeighbors+int(key)-1).values.reshape(-1)].values.reshape(shape),index=keysOfNearestNeighbors.index,columns=keysOfNearestNeighbors.columns)

			dfTGuess.loc[:,key]=(weights*y).sum(axis=1)
			
	return dfTActual,dfTGuess




		
def plotRho(dfRho,ax=None,fig=None):
	""" plotting function for rho, the correlation coefficient """
	if ax==None:
		fig,ax=_plt.subplots()
	
	for i,(key,val) in enumerate(dfRho.iteritems()):
		
		ax.plot(val,'-x',label='E=%d'%val.name)
		_finalizeSubplot(	ax,
									ylim=[0,1.02],
									xlim=[1,dfRho.index[-1]],
									xlabel=r'Prediction time steps ($\tau_p$)',
									ylabel=r'Correlation coefficient ($\rho$)',
									legendOn=True)
	return fig,ax
		
def plotFitVsActual(sFit,sActual,ax=None):
	""" plots fit data vs. actual data for a qualitative "goodness of fit" plot """
	if ax==None:
		fig,ax=_plt.subplots()
	
	
	ax.plot(sActual,sFit,'.',label=r'$\tau_p$=%s'%sFit.name,markersize=3)
	ax.plot([-1,0.5],[-1,0.5],'k')
	_finalizeSubplot(	ax,
							ylim=[-1,0.5],
							xlim=[-1,0.5],
							xlabel='Actual',
							ylabel='Fit',
							legendOn=True,
							numberLegendPoints=3
							)
	

def correlationCoefficient(data,fit):
	#TODO update to process a multi-column dataframe
	""" 
	Correlation coefficient 
	
	Reference
	---------
	https://mathworld.wolfram.com/CorrelationCoefficient.html
	"""
	if type(data)==_pd.core.frame.DataFrame or type(data)==_pd.core.frame.Series:
		y=data.values.reshape(-1)
		f=fit.values.reshape(-1)
	elif type(data)==_np.ndarray:
		y=data.reshape(-1)
		f=fit.reshape(-1)
	SSxy=((f-f.mean())*(y-y.mean())).sum()
	SSxx=((f-f.mean())**2).sum()
	SSyy=((y-y.mean())**2).sum()
	rho=SSxy**2/(SSxx*SSyy)
	return rho



	

def calcRho(dfTActual,dfTFit,E,T,plot=True):
	""" calculates correlation coefficient between Fit and Actual data """
	dfRho=_pd.DataFrame(index=range(1,T+1))
	for t in range(1,T+1):
		if plot=='all':
			plotFitVsActual(dfTFit[str(t)],dfTActual[str(t)])
		dfRho.at[t,E]=correlationCoefficient(dfTActual[str(t)],dfTFit[str(t)])
		
	if plot==True or plot=='all':
		plotRho(dfRho)
	return dfRho


def findNearestNeighbors(X,Y,numberOfNearestPoints=1):
	"""
	Find the nearest neighbors in X to each point in Y
	
	Example
	-------
	::
		
		from johnspythonlibraries2.Plot import finalizeSubplot as _finalizeSubplot

		# create data
		x=_np.arange(0,10+1)
		y=_np.arange(100,110+1)
		X,Y=_np.meshgrid(x,y)
		X=X.reshape((-1,1))
		Y=Y.reshape((-1,1))
		A=_np.concatenate((X,Y),axis=1)
		
		# points to investigate
		B=[[5.1,105.1],[8.9,102.55]]
		
		points,indices,radii=findNearestNeighbors(A,B,numberOfNearestPoints=5)
		
		for i in range(len(B)):
			fig,ax=_plt.subplots()
			ax.plot(X,Y,'.',label='original data')
			ax.plot(B[i][0],B[i][1],'x',label='point of interest')
			ax.plot(points[i][:,0],points[i][:,1],label='nearest neighbors',marker='o',linestyle='', markerfacecolor="None")
			_finalizeSubplot(ax)
			
	"""
	
	from sklearn.neighbors import NearestNeighbors

	neigh = NearestNeighbors(n_neighbors=numberOfNearestPoints)
	neigh.fit(X)
	radii,indices=neigh.kneighbors(Y)
	points=X[indices]
	
	return points, indices, radii


###################################################################################
### Main function

def main(sx,E,T,sy=None,tau=1,method='simplex',plot=False,weightingFunction='default',showTime=False):
	
#	if __name__=="__main__":
#		if False:
#			sy=None
#			tau=1
#			method='simplex'
#			plot=False
#			weightingFunction='default'
#			showTime=False
#			sx=_pd.Series(createSignal(1000,plot=False))
#		else:
#			tau=1
#			method='simplex'
#			plot=False
#			weightingFunction='default'
#			showTime=False
#			t=_np.arange(0,1000)
#			sx=_pd.Series(_np.sin(2*_np.pi*1e-1*t))
#			sy=_pd.Series(_np.sin(2*_np.pi*1e-1*t)+_np.random.uniform(-1,1,t.shape)*0.1)
#			
#		
	N=sx.shape[0]
	if method == 'simplex':
		knn=E+1
	elif method == 'smap':
		knn=sx.shape[0]//2-E+1
	
	print("N=%d, E=%d, T=%d, tau=%d, numNearestNeighbors=%d, method=%s, weighting=%s"%(N,E,T,tau,knn,method,weightingFunction))
	
	start = _time.time()
	
	if showTime: printTime('Step 1 - Test and training data sets',start)
	if type(sy)==type(None):
		sX,sY,sx=splitData(sx)
	else:
		sX=sx.copy()
		sY=sy.copy()
		
	if showTime: printTime('Step 2 - Convert to state space',start)
	dfX=convertToStateSpace(sX,E,tau)
	dfY=convertToStateSpace(sY,E,tau)
		
	if showTime: printTime('Step 3 - Find nearest neighbors',start)
	coordinates, indices, radii=findNearestNeighbors(dfX.values,dfY.values,knn)
	keysOfNearestNeighbors=_pd.DataFrame(indices+dfX.index[0],index=dfY.index)
	radii=_pd.DataFrame(radii,index=dfY.index)
		
	if showTime: printTime('step 4 - Weighting',start)
	if weightingFunction=='default':
		if method == 'smap':
			weightingFunction='exponential'
		elif method == 'simplex':
			weightingFunction='uniform'
	weights=weighting(radii,method=weightingFunction)
		
	if showTime: printTime('step 5 - Forecasting',start)
	dfTActual,dfTGuess=forecasting(sx,dfY,keysOfNearestNeighbors,weights,T=T)
				
	if showTime: printTime('step 6 - Calculate correlation coefficient, rho',start)
	dfRho=calcRho(dfTActual,dfTGuess,E=E,T=T,plot=plot)
	
	if showTime: printTime('done!',start)
	
	dic=dict({ 	'dfX':dfX,
				'dfY':dfY,
				'dfRho':dfRho,
				'dfTGuess':dfTGuess,
				'dfTActual':dfTActual,
				'keysOfNearestNeighbors':keysOfNearestNeighbors,
				'radii':radii,
				'weights':weights})
		
	return dic


###################################################################################
### functions that call main()
def determineDimensionality(sx,T,tau=1,Elist=_np.arange(1,10+1),method="simplex",weightingFunction='exponential'):
#	Elist=E
	
	# interate through each E
	dicAll=[]
	for i,E in enumerate(Elist):
		
		# perform analysis and save rho
		dic=main(sx,E,T,tau,method=method,weightingFunction=weightingFunction)
		dicAll.append(dic)
		dfRho1=dic['dfRho']
		if i==0:
			dfRho=dfRho1
		else:
			dfRho=_pd.concat((dfRho,dfRho1),axis=1)
			
	# plot rho for each E
	fig,_=plotRho(dfRho)
	ax=fig.axes[0]
	ax.set_title("N=%d, E=%d, T=%d, tau=%d, method=%s, weighting=%s"%(sx.shape[0],E,T,tau,method,weightingFunction))
#	fig.savefig('images/results_summary.png')
	return dicAll


###################################################################################
### Examples
	

def example5():
	""" work in progress """
	import xarray as xr
	import numpy as np
	Earray=np.arange(2,20)
	Tarray=np.arange(1,100)
	Tauarray=np.arange(1,2)
	da = xr.DataArray(
					  dims=['E', 'T','tau'],
	                  coords={'E': Earray,
					   'T': Tarray,
					   'tau': Tauarray,}
					   )
	def sigGen(t,phase=0):
		return _pd.Series(_np.sin(2*_np.pi*0.13e-1*t+phase))
	
	
	plot=False
	t=_np.arange(0,1001)
	syA=sigGen(t)
	sxA=syA*0.5+_np.random.uniform(-1,1,t.shape)*0.1
#	sxA=sx[:sx.shape[0]//2]
#	syA=sy[:sy.shape[0]//2]
	
	for i,E in enumerate(Earray):
		for j,T in enumerate(Tarray):
			for k,tau in enumerate(Tauarray):
				
#	E=5
#	T=10
#	tau=1
				dic=main(sxA,E=E,T=T,sy=syA,tau=tau,method='simplex',plot=False,weightingFunction='default')
				dfTActual=dic['dfTActual']
				dfTGuess=dic['dfTGuess']
				if plot:
					fig,ax=_plt.subplots()
					ax.plot(sxA)
					ax.plot(dfTGuess['1'])
					
			
				syB=sigGen(t,_np.pi/2)
				sxB=syB*0.5+_np.random.uniform(-1,1,t.shape)*0.1
				
				dfY=convertToStateSpace(sxB,E=E,tau=tau)
				
				dfTActual,dfTGuess=forecasting(sxB,dfY,dic['keysOfNearestNeighbors'],dic['weights'],T=1)
				if plot:
					fig,ax=_plt.subplots()
					ax.plot(sxB)
					ax.plot(dfTGuess)
				
				dfTGuess=dfTGuess.dropna()
				dfTActual=dfTActual.loc[dfTGuess.index]
				
				rho=correlationCoefficient(	dfTActual.values.reshape(-1),
									dfTGuess.values.reshape(-1))
	
				da.loc[E,T,tau]=rho
	
	
def example3():
	""" several sine waves with noise """
	# TODO work in progress
	
	import numpy as np
	np.random.seed(0)
	t=np.arange(0,1001)
	if True:
		noise=np.random.uniform(-1,1,len(t))*0.2
	else:
		noise=np.zeros(t.shape)
	
	def genSig(t,a,f,phi=0):
		return(a*np.sin(2*np.pi*f*t+phi))
		
	y=noise
	freqs=np.array([1.0/20,1.0/217])
#	freqs=np.array([1.0/20])
	for f in freqs:
		y+=genSig(t,1,f)
#	np.sin(2*np.pi*0.1*t)+np.sin(2*np.pi*0.007121*t)+noise
	sx=_pd.Series(y,index=t)
	
	_plt.close('all')
	fig,ax=_plt.subplots()
	_plt.plot(sx)
	
#	determineDimensionality(sx,T=200,tau=1,Elist=np.arange(1,10+1))
	
	tau=1
	dic1=main(sx,E=1,T=200,tau=tau)
	dic3=main(sx,E=3,T=200,tau=tau)
	dic5=main(sx,E=5,T=200,tau=tau)
	dic10=main(sx,E=10,T=200,tau=tau)
	_plt.figure()
	_plt.plot(dic1['dfRho'])
	_plt.plot(dic3['dfRho'])
	_plt.plot(dic5['dfRho'])
	
	dfx=_pd.DataFrame()
	dfx['Time']=np.arange(1,sx.shape[0]+1)
	dfx['x']=sx
	import pyEDM
#	pyEDM.EmbedDimension(	dataFrame=dfx,
#							lib="0 500",
#							pred="501 1000",
#							Tp=200,
#							maxE=100,
#							columns='x',
#							target='x')
	pyEDM.PredictInterval(	dataFrame=dfx,
							lib="0 500",
							pred="501 1000",
							E=10,
							maxTp=200,
							columns='x',
							target='x')
	fig=_plt.gcf()
	ax=fig.axes[0]
	ax.plot(dic10['dfRho'])
	

	
def example4():
	""" 
	lorentz attractor
	"""
	#TODO work in progress
	#TODO the peaks in this final plot appear to be shifting to the left with increasing E and tau.  As best I can tell, pyEDM has the same issue.  What does this mean?
	
	def ODEs(t,y,*args):
		X,Y,Z = y
		sigma,b,r=args
		derivs=	[	sigma*(Y-X),
					-X*Z+r*X-Y,
					X*Y-b*Z]
		return derivs
	
	## solve
	from scipy.integrate import solve_ivp
	args=[10,8/3.,28] # sigma, b, r
	T=100
	dt=0.05 #0.05 max
	psoln = solve_ivp(	ODEs,
						[0,T],
						[-9.38131377, -8.42655716 , 29.30738524],
#						[1,1,1],
						args=args,
						t_eval=_np.arange(0,T+dt,dt)
						)
	
#	_plt.close('all')
	fig,ax=_plt.subplots(3,sharex=True)
#	t=psoln.t
	x,y,z=psoln.y
	ax[0].plot(x,label='x')
	ax[1].plot(y,label='y')
	ax[2].plot(z,label='z')
	
	_plt.figure();_plt.plot(x,y)
	_plt.figure();_plt.plot(y,z)
	_plt.figure();_plt.plot(z,x)
	
	from mpl_toolkits.mplot3d import Axes3D
	fig = _plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x,y,zs=z)
	
	sx=_pd.Series(x,_np.arange(0,len(x)))
	sy=_pd.Series(y,_np.arange(0,len(x)))
	sz=_pd.Series(z,_np.arange(0,len(x)))
	
	T=500
	tau=5	
	N=sx.shape[0]
	dicAll=determineDimensionality(sx,T=T,tau=tau,Elist=_np.arange(1,10+1))
	fig=_plt.gcf()
	fig.savefig('N_%d_T_%d_tau_%d.png'%(N,T,tau))
	
	

	

def example2():
	
	""" 
	Based loosely on Sugihara 1990 paper.
	This is intended to show some exmaples of how forecasting work.
	"""
	_plt.close('all')
	
	sx=createSignal(100,plot=False)
	T=4
	E=3
	method="simplex"
	weightingFunction='exponential'
	dic=main(sx,E,T,method=method,weightingFunction=weightingFunction)
	dfY=dic['dfY']
	dfX=dic['dfX']
	weights=dic['weights']
	keysOfNearestNeighbors=dic['keysOfNearestNeighbors']
	fig,axAll=_plt.subplots(T,sharex=True)
	index=59
	for k,ax in enumerate(axAll):
		ax.plot(dfY.loc[index].index.values+index,dfY.loc[index],label='pattern to forecast',linewidth=5,color='tab:blue')
		for i,key in enumerate(keysOfNearestNeighbors.loc[index]):
			label1=''
			label2=''
			if i==0:
				label1='nearest neighbors'
				label2='Past similar points'
			ax.plot(dfX.loc[key].index.values+key,dfX.loc[key],color='tab:orange',label=label1,linewidth=5)
			ax.scatter(key+1+k,sx.loc[key+1+k],marker='o',color='orange',facecolor='none',label=label2,linewidths=2,s=50)
		
		ax.plot(sx.loc[:index],label='orig. data',color='k',marker='.',markersize=4)
		
		ypredict=[]
		x=[]
		for t in range(1,k+2):
		#	y=_pd.DataFrame(sx.loc[(keysOfNearestNeighbors.loc[index]+int(key)-1).values.reshape(-1)].values.reshape(shape),index=keysOfNearestNeighbors.index,columns=keysOfNearestNeighbors.columns)
		
			y=sx.loc[keysOfNearestNeighbors.loc[index].values+t].values
			w=weights.loc[index].values
			ypredict.append((y*w).sum())
			x.append(index+t)
			
		ax.scatter([index+t],[sx.loc[index+t]],marker='o',color='b',facecolor='none',label='next point to predict',s=50)
		ax.scatter(x[-1],ypredict[-1],marker='x',color='green',label='next predicted value')
	
		ax.plot(	_np.concatenate(([index],x)),
					_np.concatenate(([sx.loc[index]],ypredict)),
					linestyle='--',
					color='green',
					label='predicted',
#						marker='x',
					ms=3
					)
		
		
			
		_finalizeSubplot(ax,
						  xlim=[0,index+5],
						  subtitle='T=%d'%(k+1),
						  legendOn=False)
		if k==0:
			_legendOutside(ax)
	axAll[0].set_title('E=%d, %s method, %s weighting'%(E,method,weightingFunction))
	ax.set_xlabel('Time')
	for i in range(2):
		_finalizeFigure(fig,h_pad=0.5,)


def example1():
	""" 
	Sugihara 1990 paper.
	Effectively reproduces one of the figures in this paper and provides
	several supplementary plots
	"""
	
	
	## initialize
	_plt.close('all')
	sx=createSignal(1000,plot=False)
	start = _time.time()
	printTime('Starting',start)
	
	# parameters		
	T=10						# steps into time to predict
	tau =1		
	Elist=_np.arange(1,10+1) 	# embedded dimension
	method="simplex"
	weightingFunction='exponential'
	
	# interate through each E
	for i,E in enumerate(Elist):
		
		# perform analysis and save rho
		dic=main(sx,E,T,tau,method=method,weightingFunction=weightingFunction)
		dfRho1=dic['dfRho']
		dfTGuess=dic['dfTGuess']
		dfTActual=dic['dfTActual']
		if i==0:
			dfRho=dfRho1
		else:
			dfRho=_pd.concat((dfRho,dfRho1),axis=1)
			
		# optional plots
		if E==3:
			fig,ax=_plt.subplots(2,2)
			ax[0][0].plot(sx,linewidth=0.4)
			_finalizeSubplot(ax[0][0],
									xlabel='Time (t)',
									ylabel=r'$\Delta x(t)$',
									legendOn=False)
			plotFitVsActual(dfTGuess['2'],dfTActual['2'],ax[0][1])
			plotFitVsActual(dfTGuess['5'],dfTActual['5'],ax[1][0])
			plotRho(_pd.DataFrame(dfRho[E]),ax[1][1])
			_finalizeFigure(fig,h_pad=3,w_pad=3,pad=1)
#			fig.savefig('images/E3_figure.png')
			
			fig,ax=_plt.subplots()
			index=700
			ax.plot(sx,linewidth=5,color='grey',alpha=0.4,label='Original Data')
			ax.plot(dfTGuess.columns.values.astype(int)+index,
				   dfTGuess.loc[index+1],
				   color='tab:orange',
				   label='Forecast, t=%d'%index)
			_finalizeSubplot(ax,
							   xlabel='Time (t)',
							   ylabel=r'$\Delta x(t)$',
							   xlim=[index-4,index+4+10],
							   ylim=[-1,0.5],
							   title='Example forecast, E=%d'%E)
#			fig.savefig('images/example_forecast.png')
		
		printTime('',start)
	
	# plot rho for each E
	fig,_=plotRho(dfRho,fig=fig)
#	fig.savefig('images/results_summary.png')
	
	printTime('Done!',start)