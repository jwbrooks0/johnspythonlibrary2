"""

Empirical dynamic modelling (EDM) toolkit

Work in progress

"""

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import time as _time
from deprecated import deprecated
from johnspythonlibrary2.Plot import finalizeSubplot as _finalizeSubplot
from johnspythonlibrary2.Plot import finalizeFigure as _finalizeFigure
from johnspythonlibrary2.Plot import legendOutside as _legendOutside
from johnspythonlibrary2.Plot import heatmap as _heatmap
from johnspythonlibrary2.Plot import subTitle as _subtitle

###################################################################################
#%% signal generation
# various generated signals to test code in this library


def twoSpeciesWithBidirectionalCausality(N,tau_d=0,IC=[0.2,0.4],plot=False,params={'Ax':3.78,'Ay':3.77,'Bxy':0.07,'Byx':0.08}):
	"""
	Coupled two equation system with bi-directional causality.  
	
	Eq. 1 in Ye 2015

	Reference
	---------
	Eq. 1 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4592974/
	
	
	Examples
	--------
	
	::
	
		N=3000
		twoSpeciesWithBidirectionalCausality(N,plot=True)
		twoSpeciesWithBidirectionalCausality(N,tau_d=2,plot=True)

	"""
	
# 	N+=tau_d
	
	x=_np.ones(N+tau_d,dtype=float)*IC[0]
	y=_np.ones(N+tau_d,dtype=float)*IC[1]
	
# 	for i in range(tau_d,N+tau_d-1):
# 		x[i+1]=x[i]*(3.78-3.78*x[i]-0.07*y[i])
# 		y[i+1]=y[i]*(3.77-3.77*y[i]-0.08*x[i-tau_d])
		
	for i in range(tau_d,N+tau_d-1):
		x[i+1]=x[i]*(params['Ax']-params['Ax']*x[i]-params['Bxy']*y[i])
		y[i+1]=y[i]*(params['Ay']-params['Ay']*y[i]-params['Byx']*x[i-tau_d])
		
	x=x[tau_d:]
	y=y[tau_d:]
		
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(x)
		ax.plot(y)
		
	return _pd.Series(x),_pd.Series(y)



def predatorPrey():
	return 'work in progress'
	# TODO add predator-prey model

def createTentMap(N=1000,plot=False,x0=_np.sqrt(2)/2.0):
	""" generate fake data using a tentmap """
	t=_np.arange(0,N+1)
	x=_np.zeros(t.shape,dtype=_np.float128)
	x[0]=x0
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


def solveLorentz(	N=2000,
					dt=0.05,
					IC=[-9.38131377, -8.42655716 , 29.30738524],
					addNoise=False,
					plot=False,
					seed=[0,1,2],
					T=1):
	"""
	Solves the lorentz attractor nonlinear ODEs.
	
	References
	----------
	 * https://en.wikipedia.org/wiki/Lorenz_system
	"""
	T=N*dt
	
	from scipy.integrate import solve_ivp
		
	def ODEs(t,y,*args):
		X,Y,Z = y
		sigma,b,r=args
		derivs=	[	sigma*(Y-X),
					-X*Z+r*X-Y,
					X*Y-b*Z]
		return derivs

	args=[10,8/3.,28] # sigma, b, r
	
# 	print(T)
	t_eval=_np.arange(0,T+dt,dt)
# 	print(t_eval)
	psoln = solve_ivp(	ODEs,
						[0,T],
						IC,  # initial conditions
						args=args,
						t_eval=t_eval
						)
	
	x,y,z=psoln.y
	if addNoise==True:
		noiseAmp=5
		_np.random.seed(seed[0])
		x+=_np.random.rand(x.shape[0])*noiseAmp
		_np.random.seed(seed[1])
		y+=_np.random.rand(y.shape[0])*noiseAmp
		_np.random.seed(seed[2])
		z+=_np.random.rand(z.shape[0])*noiseAmp
			
		
	if plot!=False:
		fig,ax=_plt.subplots(3,sharex=True)
		markersize=2
		ax[0].plot(x,marker='.',markersize=markersize);ax[0].set_ylabel('x')
		ax[0].set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args[0],args[1],args[2])+'\nIC = (%.3f, %.3f, %.3f)'%(IC[0],IC[1],IC[2]))
		ax[1].plot(y,marker='.',markersize=markersize);ax[1].set_ylabel('y')
		ax[2].plot(z,marker='.',markersize=markersize);ax[2].set_ylabel('z')
		ax[2].set_xlabel('Time')
		
	if plot=='all':
		_plt.figure();_plt.plot(x,y)
		_plt.figure();_plt.plot(y,z)
		_plt.figure();_plt.plot(z,x)
	
		from mpl_toolkits.mplot3d import Axes3D
		fig = _plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(x,y,zs=z)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')		
		ax.set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args[0],args[1],args[2])+'\nIC = (%.3f, %.3f, %.3f)'%(IC[0],IC[1],IC[2]))
		
		
	return x,y,z


def coupledHarmonicOscillator(	N=10000,
								  T=1,
								IC=[0,0.9,0,-1],
								args=[1,1,1e-4],
								plot=False):
	"""
	#TODO
	

	http://users.physics.harvard.edu/~schwartz/15cFiles/Lecture3-Coupled-Oscillators.pdf

	"""
	
	import matplotlib.pyplot as _plt
	
# 	N=10000
# 	T=1
	dt=T/N
# 	IC=[0,0.9,0,-1]
# 	args=[1,1,1e-4]
# 	plot=True
# 	
	
	if False:
		k,kappa,m=args
		omega_s=_np.sqrt(k/m)
		omega_f=_np.sqrt((k+2*kappa)/m)
	
	from scipy.integrate import solve_ivp
	 
	 
# 	T=N*dt
		
	def ODEs(t,y,*args):
		y1,x1,y2,x2 = y
		k,kappa,m=args
		derivs=	[	-(k+kappa)/m*x1+kappa/m*x2,
					y1,  
 					-(k+kappa)/m*x2+kappa/m*x1,
					y2]
		return derivs

	time=_np.arange(0,T,dt)
	psoln = solve_ivp(	ODEs,
					t_span=[0,T],
					y0=IC,  # initial conditions
					args=args,
					t_eval=time
					)
	
	y1,x1,y2,x2 =psoln.y
	
	if plot==True:
		fig,ax=_plt.subplots(2,sharex=True)
		ax[0].plot(time,x1,marker='.');ax[0].set_ylabel('x1')
		ax[0].set_title('Coupled harmonic oscillator\n'+r'(k, kappa, m)='+'(%.3f, %.3f, %.6f)'%(args[0],args[1],args[2])+'\nIC = (%.3f, %.3f, %.3f, %.3f)'%(IC[0],IC[1],IC[2],IC[3]))
		ax[1].plot(time,x2,marker='.');ax[1].set_ylabel('x2')
		
	s1=_pd.Series(x1,index=time)
	s2=_pd.Series(x2,index=time)
	return s1,s2


###################################################################################
#%% plotting functions


		
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
	
	
def correlationHeatmap(x,y,Z,xlabel='',ylabel='',showMax=True,cbarMinMax=None):
	import numpy as _np
	import seaborn as sb
	# from matplotlib.colors import LogNorm
	fig,ax=_plt.subplots()
	
	
	if type(cbarMinMax) != type(None):
		vmin=cbarMinMax[0]
		vmax=cbarMinMax[1]
	else:
		vmin=_np.nanmin(Z)
		vmax=_np.nanmax(Z)
	ax=sb.heatmap(	Z,
 			vmin=vmin,
 			vmax=vmax,
# 			norm=LogNorm(vmin=0.01, vmax=1),
			)
	
	ax.collections[0].colorbar.set_label("Pearson correlation")
	
	ax.invert_yaxis()
	
	ax=_plt.gca()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
# 	xticks=ax.get_xticks()

	if showMax==True:
		
		import matplotlib as _mpl
		_mpl.rcParams['legend.numpoints'] = 1
		
		def coordinate_map(in_data,coordinate='x',):
			""" 
			Note that the ticks and ticklabels are in heatmap() do not have a 1 to 1 mapping.
			this function determines the mapping for the x and y coordinates and applies it to whatever data is provided.
			"""
			if coordinate=='x':
				index=0
				ticklabels_obj=ax.get_xticklabels()
			elif coordinate=='y':
				index=1
				ticklabels_obj=ax.get_yticklabels()
				
			ticks=_np.zeros(len(ticklabels_obj),dtype=float)
			ticklabels=_np.zeros(len(ticklabels_obj),dtype=float)
			
			for i in range(len(ticklabels_obj)):
				ticks[i]=ticklabels_obj[i].get_position()[index]
				ticklabels[i]=float(ticklabels_obj[i].get_text())
				
			slope,intercept=_np.polyfit(ticklabels,ticks,deg=1)
			out_data=in_data*slope+intercept
			return out_data
		
		
		ymax,xmax=_np.where(Z.max().max()==Z)
		xmax=Z.columns[xmax].values
		ymax=Z.index[ymax].values
		xmax_map=coordinate_map(xmax,'x')
		ymax_map=coordinate_map(ymax,'y')
		print(xmax_map,ymax_map)
		ax.plot(xmax_map,ymax_map,c='g',marker='*',label='%s_max'%ylabel,linestyle='')
# 		ax.plot(xmax_map,ymax_map,c='g',marker='*',label='E_max',linestyle='')
# 		ax.plot(xmax,ymax,'go')
		ax.set_title('Max: (%s, %s)=(%d,%d)'%(xlabel,ylabel,xmax,ymax))
		
# 		for i,(key,val) in enumerate(Z.iterrows()):
# 			print(i,key)
# 			val.ixd
		temp=Z.idxmax()
		y=temp.values
		x=temp.index.values
		
		ax.plot(coordinate_map(x,'x'),coordinate_map(y,'y'),c='g',linestyle='-',label='%s_max(%s)'%(ylabel,xlabel))
# 		ax.plot(coordinate_map(x,'x'),coordinate_map(y,'y'),c='g',linestyle='-',label='E_max(tau)')

		temp=Z.idxmax(axis=1)
		x=temp.values
		y=temp.index.values
		
		ax.plot(coordinate_map(x,'x'),coordinate_map(y,'y'),c='deepskyblue',linestyle='-',label='%s_max(%s)'%(xlabel,ylabel))



		ax.legend()
# 	ax.plot(ymax+y[0],xmax+x[0],'rx')
	
# 	return fig,ax
# 	


def dimensionHeatmap(Z,xlabel='',ylabel='',showMax=True):
	import numpy as _np
	import seaborn as sb
	from matplotlib.colors import LogNorm
	fig,ax=_plt.subplots()
	x=Z.index.values
	y=Z.columns.values
	
# 	sb.heatmap(	Z,
# # 			vmin=0,
# # 			vmax=1,
# 			norm=LogNorm(vmin=0.01, vmax=1))
	sb.heatmap(	Z,
 			vmin=0,
 			vmax=1,
# 			norm=LogNorm(vmin=0.01, vmax=1),
			)
	
	ax.invert_yaxis()
	
# 	ax=_plt.gca()
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
# 	xticks=ax.get_xticks()

	if showMax==True:
		
		import matplotlib as _mpl
		_mpl.rcParams['legend.numpoints'] = 1
		
		def coordinate_map(in_data,coordinate='x',):
			""" 
			Note that the ticks and ticklabels are in heatmap() do not have a 1 to 1 mapping.
			this function determines the mapping for the x and y coordinates and applies it to whatever data is provided.
			"""
			if coordinate=='x':
				index=0
				ticklabels_obj=ax.get_xticklabels()
			elif coordinate=='y':
				index=1
				ticklabels_obj=ax.get_yticklabels()
				
			ticks=_np.zeros(len(ticklabels_obj),dtype=float)
			ticklabels=_np.zeros(len(ticklabels_obj),dtype=float)
			
			for i in range(len(ticklabels_obj)):
				ticks[i]=ticklabels_obj[i].get_position()[index]
				ticklabels[i]=float(ticklabels_obj[i].get_text())
				
			slope,intercept=_np.polyfit(ticklabels,ticks,deg=1)
			out_data=in_data*slope+intercept
			return out_data
		
		
		ymax,xmax=_np.where(Z.max().max()==Z)
		xmax=Z.columns[xmax].values
		ymax=Z.index[ymax].values
		xmax_map=coordinate_map(xmax,'x')
		ymax_map=coordinate_map(ymax,'y')
		print(xmax_map,ymax_map)
		ax.plot(xmax_map,ymax_map,c='g',marker='*',label='E_max',linestyle='')
# 		ax.plot(xmax,ymax,'go')
		ax.set_title('Max: (%s, %s)=(%d,%d)'%(xlabel,ylabel,xmax,ymax))
		
# 		for i,(key,val) in enumerate(Z.iterrows()):
# 			print(i,key)
# 			val.ixd
		temp=Z.idxmax()
		y=temp.values
		x=temp.index.values
		
		ax.plot(coordinate_map(x,'x'),coordinate_map(y,'y'),c='g',linestyle='-',label='E_max(T)')
		ax.legend()


###################################################################################
#%% sub-functions
def printTime(string,start):
	""" print time since start """
	if string!='':
		print('%s'%(string))
	print('%.3fs'%((_time.time()-start)))
	
	
# @deprecated
# def forecast_old(sx,dfY,keysOfNearestNeighbors,weights,T,plot=False):
# 	""" The forecasting method.  Combines weights with correct indices (keys) to get the forecast """
# 	# TODO this file REQUIRES dfY as an input.  Make this optional
# 	# TODO I believe that this function is the bottleneck for this algorithm.  Try to optomize
# 	# TODO write a case for T=0
# 	s=sx.copy()
# 	dfTActual=_pd.DataFrame(index=dfY.index.values[:-T+1],dtype=float)
# 	for key in range(1,1+T):
# 		dfTActual['%d'%key]=dfY.iloc[key-1:dfY.shape[0]-T+key][0].values	
# 	dfTGuess=_pd.DataFrame(index=dfTActual.index,columns=dfTActual.columns,dtype=float)
# 	
# 	for key,val in dfTGuess.iteritems():
# 			print(key)
# 			shape=keysOfNearestNeighbors.shape
# 			y=_pd.DataFrame(sx.loc[(keysOfNearestNeighbors+int(key)-1).values.reshape(-1)].values.reshape(shape),index=keysOfNearestNeighbors.index,columns=keysOfNearestNeighbors.columns)

# 			dfTGuess.at[:,key]=(weights*y).sum(axis=1).values
# 		
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		_plt.plot(dfTActual['1'],label='actual')
# 		_plt.plot(dfTGuess['1'],label='guess')
# 		_plt.legend()
# 	return dfTActual,dfTGuess


def applyForecast(s,dfY,keys,weights,T,plot=False):
	""" 
	The forecasting method.  Combines weights with correct indices (keys) to get the forecast 
	
	Examples
	--------
	Example 1::
		
		import numpy as np
		import matplotlib.pyplot as plt; plt.close('all')
		import pandas as pd
		
		N=500
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=3
		knn=E+1
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		keys,weights=createMap(Px,Py,knn=knn)
		
		dfTActual, dfTGuess=applyForecast(s,Py,keys,weights,T=10,plot=True)
	
	"""
	# TODO update so that T starts at 0 instead of 1
	
	index=dfY.index.values[:-T]
	dfTActual=_pd.DataFrame(index=index,dtype=float)
	for key in range(0,1+T):
		dfTActual[key]=s.loc[index+key].values	
	dfTGuess=_pd.DataFrame(index=dfTActual.index,columns=dfTActual.columns,dtype=float)
	
	for key,val in dfTGuess.iteritems():
			shape=keys.loc[index].shape
# 			y=s.loc[keys.loc[index+key].values.reshape(-1)].values.reshape(shape)
			y=s.loc[keys.loc[index].values.reshape(-1)+key].values.reshape(shape)

			dfTGuess.at[:,key]=(weights.loc[index]*y).sum(axis=1).values
		
	if plot==True:
		
		fig,ax=_plt.subplots(T,sharex=True,sharey=True)
		rho=calcCorrelationCoefficient(dfTActual, dfTGuess)
		for i, Ti in enumerate(range(1,1+T)):
			ax[i].plot(dfTActual[Ti].index.values+Ti,dfTActual[Ti],label='actual')
			ax[i].plot(dfTGuess[Ti].index.values+Ti,dfTGuess[Ti],label='guess')
# 			ax[i].legend()
			_subtitle(ax[i],'T=%d, rho=%.3f'%(Ti,rho[i]))
			_finalizeSubplot(ax[i],legendOn=False,xlim=[s.index[-1]//2,s.index[-1]+1])
		_finalizeSubplot(ax[-1],legendOn=False,xlim=[s.index[-1]//2,s.index[-1]+1],xlabel='Time')
		ax[0].set_title('N=%d'%len(s))
		_finalizeFigure(fig,h_pad=0)
	return dfTActual,dfTGuess



def reconstruct(sx,keys,weights):
	"""
	
	Examples
	--------
	Example 1::
		
		N=100
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=3
		knn=E+1
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		keys,weights=createMap(Px,Py,knn=knn)
		v=reconstruct(sx,keys,weights)
		
		index=55
# 		E=keys.shape[1]
		plt.figure()
		plt.plot(sx,'k',label='Training data')
		plt.plot(sy,'b',label='Test data')
		i=np.arange(index-E+1,index+1)
		plt.plot(i,sy.loc[i],'r',marker='',linewidth=2)
		plt.plot(index,sy.loc[index],'r',marker='x',label='Point in question')
		count=0
		for j in keys.loc[index]:
			print(j)
			if count==0:
				label='Nearest neighbors'
			else:
				label=''
			
			i=np.arange(j-E+1,j+1)
			plt.plot(i,sx.loc[i],'g',marker='',linewidth=2)
			plt.plot(j,sx.loc[j],'g',marker='.',label=label)
			count+=1
			
			
		i=np.arange(index-E+1,index+1)
	# 	plt.plot(i,v.loc[i],'m',marker='',linewidth=2)
		plt.plot(index,v.loc[index],'m',marker='o',mfc='none',
				label='Reconstruction',linestyle='')
	# 	plt.plot()
	
# 		plt.plot(v,'m',label='Reconstruction')
		plt.legend()
		plt.xlim(0,60)
		plt.title('Reconstruction, single pt.\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.savefig('reconstruction_N_%d_E_%d_tau_%d.png'%(N,E,tau))
		
		
		plt.figure()
		plt.plot(sx,'k',label='Training data')
		plt.plot(sy,'b',label='Test data')
		plt.plot(v,'m',label='Reconstruction',linestyle='--')
		plt.legend()
		plt.title('Reconstruction, all\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.xlim(45,100)
		plt.savefig('reconstruction2_N_%d_E_%d_tau_%d.png'%(N,E,tau))
		
	Example 2::
		
		
		N=200
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=4
		knn=E+1
		tau=2
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		keys,weights=createMap(Px,Py,knn=knn)
		v=reconstruct(sx,keys,weights)
		
		index=54*2
		plt.figure()
		plt.plot(sx,'k',label='Training data')
		plt.plot(sy,'b',label='Test data')
		i=np.arange(index-((E-1)*tau),index+1,tau)
		plt.plot(i,sy.loc[i],'r',marker='',linewidth=2)
		plt.plot(index,sy.loc[index],'r',marker='x',label='Point in question')
		count=0
		for j in keys.loc[index]:
			print(j)
			if count==0:
				label='Nearest neighbors'
			else:
				label=''
			
			i=np.arange(j-((E-1)*tau),j+1,tau)
			plt.plot(i,sx.loc[i],'g',marker='',linewidth=2)
			plt.plot(j,sx.loc[j],'g',marker='.',label=label)
			count+=1
			
			
		i=np.arange(index-((E-1)*tau),index+1,tau)
		plt.plot(index,v.loc[index],'m',marker='o',mfc='none',
				label='Reconstruction',linestyle='')
		plt.legend()
		plt.xlim(0,110)
		plt.title('Reconstruction, single pt.\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.savefig('reconstruction_N_%d_E_%d_tau_%d.png'%(N,E,tau))
		
		
		plt.figure()
		plt.plot(sx,'k',label='Training data')
		plt.plot(sy,'b',label='Test data')
		plt.plot(v,'m',label='Reconstruction',linestyle='--')
		plt.legend()
		plt.title('Reconstruction, all\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.xlim(90,200)
		plt.savefig('reconstruction2_N_%d_E_%d_tau_%d.png'%(N,E,tau))
		
		
	Example 3::
		
		## reconstruction using only one set of data (i.e. creating a map to itself)
		
		N=100
		s=createTentMap(N=N)
		
		E=4
		knn=E+1
		tau=1
		
		P=convertToStateSpace(s,E=E,tau=tau)[0]
		
		keys,weights=createMap(P,P,knn=knn)
		v=reconstruct(s,keys,weights)
		
		index=71
		
		plt.figure()
		plt.plot(s,'k',label='Data')
		i=np.arange(index-E+1,index+1)
		plt.plot(i,s.loc[i],'r',marker='',linewidth=2)
		plt.plot(index,s.loc[index],'r',marker='x',label='Point in question')
		
		count=0
		for k,j in enumerate(keys.loc[index]):
			if k==0:
				continue
			
			print(j)
			if count==0:
				label='Nearest neighbors'
			else:
				label=''
			
			i=np.arange(j-E+1,j+1)
			plt.plot(i,s.loc[i],'g',marker='',linewidth=2)
			plt.plot(j,s.loc[j],'g',marker='.',label=label)
			count+=1
			
			
		i=np.arange(index-E+1,index+1)
		plt.plot(index,v.loc[index],'m',marker='o',mfc='none',
				label='Reconstruction',linestyle='')
	
		plt.legend()
		plt.title('Reconstruction, single pt.\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.savefig('reconstruction_N_%d_E_%d_tau_%d_sameData.png'%(N,E,tau))
		
		
		plt.figure()
		plt.plot(s,'k',label='Data')
		plt.plot(v,'m',label='Reconstruction',linestyle='--')
		plt.legend()
		plt.title('Reconstruction, all\nN=%d, E=%d, tau=%d'%(N,E,tau))
		plt.savefig('reconstruction2_N_%d_E_%d_tau_%d_sameData.png'%(N,E,tau))
		
		
	Example 4 ::
		
		import matplotlib.pyplot as plt
		
		N=2000-1
		x,y,z=solveLorentz(N=N)
		s1A=[_pd.Series(x[:1000]),_pd.Series(y[:1000])]
		s1B=[_pd.Series(x[1000:]),_pd.Series(y[1000:])]
		s2A=_pd.Series(z[:1000])
		s2B=_pd.Series(z[1000:])
		
		E=3
		tau=1
		
		P1A=convertToTimeLaggedSpace(s1A, E=E, tau=tau)
		P1B=convertToTimeLaggedSpace(s1B, E=E, tau=tau)
		
		keys,weights = createMap(P1A,P1B,knn=2*E+1)
		
		s2B_recon=reconstruct(s2A,keys,weights)
				
		plt.figure()
		plt.plot(s2B,'k',label='Data')
		plt.plot(s2B_recon,'m',label='Reconstruction',linestyle='--')
		plt.legend()
		plt.title('Reconstruction, all\nN=%d, E=%d, tau=%d'%(N,E,tau))

		
		
	"""
	index=keys.index
	shape=keys.shape
	s=_pd.DataFrame(sx[keys.values.reshape(-1)].values.reshape(shape),
				  index=index)
	v=(s*weights).sum(axis=1)
	
	return v


def createMap(PA1,PA2,knn,weightingMethod='exponential'):
	"""
	
	Examples
	--------
	Example 1::
			
			
			
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd
		
		N=100
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=3
		knn=E+1
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		keys,weights=createMap(Px,Py,knn=knn)
		
		index=N//2+5
		plt.figure()
		plt.plot(sx,label='Training data',color='k')
		plt.plot([sx.index[-1],sx.index[-1]+1],[sx.iloc[-1],sy.iloc[0]],color='k')
		plt.plot(sy,label='Test data',color='blue')
		plt.plot(Py.loc[index].index.values+index,Py.loc[index].values,'r',marker='x',label='Points in question',linewidth=2)
		for j,i in enumerate(keys.loc[index]):
			print(i)
			if j==0:
				label='nearest neighbors'
			else:
				label=''
			plt.plot(Px.loc[i].index.values+i,Px.loc[i].values,'g',marker='.',label=label,linewidth=2)
	
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(Px[-2].values,Px[-1].values,Px[0].values,linestyle='',marker='.')
		ax.plot([Py.loc[index,-2]],[Py.loc[index,-1]],[Py.loc[index,-0]],linestyle='',marker='x',color='r')
		
		coordinates, indices, radii=findNearestNeighbors(Px.values,Py.values,numberOfNearestPoints=knn)
		i=np.where(Py.index==index)[0][0]
		ax.plot(coordinates[i,:,0],coordinates[i,:,1],coordinates[i,:,2],linestyle='',marker='o',mfc='none',color='g')
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_zlabel('x3')
		

# 		soln=[]
# 		for i,ind in enumerate(keys.loc[index]):
# 			print(ind)
# 			soln.append(Px.loc[ind].values)
# 		soln=np.array(soln)
# 		w=weights.loc[index]
# 		reconstruction=np.matmul(w,soln)
# 		plt.plot(Py.loc[index].index.values+index,reconstruction,color='m',linestyle='--',marker='o',label='reconstruction',linewidth=3,mfc='none')
	
		plt.title('Nearest neighbors: N=%d, E=%d, tau=%d'%(N,E,tau))
		plt.xlim(0,60)
		plt.legend()
		plt.savefig('createMap_N_%d_E_%d_tau_%d.png'%(N,E,tau))
		
# 		plt.plot(reconstruct(sx,keys,weights),color='orange')
		
		
	Example 2 ::
		
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd
		
		N=200
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
		E=4
		knn=E+1
		tau=2
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		keys,weights=createMap(Px,Py,knn=knn)
		
		index=54*2
		plt.figure()
		plt.plot(sx,label='Training data',color='k')
		plt.plot([sx.index[-1],sx.index[-1]+1],[sx.iloc[-1],sy.iloc[0]],color='k')
		plt.plot(sy,label='Test data',color='blue')
		plt.plot(Py.loc[index].index.values+index,Py.loc[index].values,'r',marker='x',label='Points in question',linewidth=2)
		for j,i in enumerate(keys.loc[index]):
			print(i)
			if j==0:
				label='nearest neighbors'
			else:
				label=''
			plt.plot(Px.loc[i].index.values+i,Px.loc[i].values,'g',marker='.',label=label,linewidth=2)
	
		plt.title('Nearest neighbors: N=%d, E=%d, tau=%d'%(N,E,tau))
		plt.xlim(0,110)
		plt.legend()
		plt.savefig('createMap_N_%d_E_%d_tau_%d.png'%(N,E,tau))
	
	
	Example 3 ::
		
		## Example where the map is created between the same data set.
		
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd
		
		N=100
		s=createTentMap(N=N)
		
		E=4
		knn=E+1
		tau=1
		
		P=convertToStateSpace(s,E=E,tau=tau)[0]
		
		keys,weights=createMap(P,P,knn=knn)
		
		index=44*2
		plt.figure()
		plt.plot(s,label='Data',color='k')
		plt.plot(P.loc[index].index.values+index,P.loc[index].values,'r',marker='x',label='Points in question',linewidth=2)
		for j,i in enumerate(keys.loc[index]):
			if j==0:
				continue
			print(i)
			if j==0:
				label='nearest neighbors'
			else:
				label=''
			plt.plot(P.loc[i].index.values+i,P.loc[i].values,'g',marker='.',label=label,linewidth=2)
	
		plt.title('Self map.  Nearest neighbors: N=%d, E=%d, tau=%d'%(N,E,tau))
		plt.xlim(-1,N)
		plt.legend()
		plt.savefig('createMap_N_%d_E_%d_tau_%d_selfMap.png'%(N,E,tau))
	
	
	Example 4 ::
		
		N=2000-1
		x,y,z=solveLorentz(N=N,plot='all')
		s1A=[_pd.Series(x[:1000]),_pd.Series(y[:1000])]
		s1B=[_pd.Series(x[1000:]),_pd.Series(y[1000:])]
		
		E=3
		tau=1
		
		P1A=convertToTimeLaggedSpace(s1A, E=E, tau=tau)
		P1B=convertToTimeLaggedSpace(s1B, E=E, tau=tau)
		
		keys,weights = createMap(P1A,P1B,knn=2*E+1)
	
	"""
	
	coordinates, indices, radii=findNearestNeighbors(PA1.values,PA2.values,numberOfNearestPoints=knn)
	keysOfNearestNeighbors=_pd.DataFrame(indices+PA1.index[0],index=PA2.index)
	radii=_pd.DataFrame(radii,index=PA2.index)
	
	# temporary code.  if a perfect match occurs (i.e. radii=0), then an error will occur.  This should take care of that.  
	radii[radii==0]=_np.nan
	
	weights=calcWeights(radii,method=weightingMethod)
	
	return keysOfNearestNeighbors,weights
	



def splitData(s):
	""" 
	split data, s, in half 
	
	
	Examples
	--------
	Example 1::
		
		s_in=pd.Series(np.arange(100))
		sX,sY,s_out=splitData(s_in)
		
	Example 2::
		
		s_in=np.arange(101)
		sX,sY,s_out=splitData(s_in)
	"""
	if type(s) == _np.ndarray:
		s=_pd.Series(s)
	elif type(s) == _pd.core.series.Series:
		pass
	else:
		raise Exception('Improper data type of input')
		
	if _np.mod(s.shape[0],2)==1:
		s=s.iloc[:-1]
	s.index=_np.arange(0,s.shape[0])
	sX=s.iloc[0:s.shape[0]//2]
	sY=s.iloc[s.shape[0]//2:]
	return sX,sY,s


@deprecated
def convertToStateSpace(s,E,tau):
	""" 
	Convert input to state space using the embedded dimension, E,
	and time step, tau 
	
	Example
	-------
	Example 1::
		
		s=_pd.Series(_np.arange(0,100))
		P=convertToStateSpace(s, E=5, tau=1)[0]
		
	Example 2::
		
		s1=_pd.Series(_np.arange(0,100))
		s2=_pd.Series(_np.arange(1000,1100))
		P1,P2=convertToStateSpace([s1,s2], E=5, tau=1)
		
	Example 3::
		
		s=_pd.Series(_np.arange(0,100))
		P=convertToStateSpace(s, E=5, tau=5)[0]
		
	Example 4::
		
		
		
# 		#s=createTentMap(300,plot=False)

		N=1000
		x,y,z=solveLorentz(N=N,plot='all')
		s=pd.Series(x)
		
		P=convertToStateSpace(s, E=2, tau=1)[0]
		
		fig,ax=plt.subplots()
		ax.plot(s,marker='.')
		
		fig,ax=plt.subplots()
		ax.plot(P.iloc[:,0],P.iloc[:,1],'.',markersize=2)
		ax.set_xlabel('Dimension 1')
		ax.set_ylabel('Dimension 2')
		
	"""
	if type(s) != list:
		s=[s]
	out=[]
	for si in s:
		index=si.index.values[(E-1)*tau:]
		columns=_np.arange(-(E-1)*tau,1,tau)
		dfs=_pd.DataFrame(index=index,columns=columns)	
		for key,val in dfs.iteritems():
			if key==0:
				dfs[key]=si.iloc[key-columns.min():si.shape[0]+key].values	
			else:
				dfs[key]=si.iloc[key-columns.min():si.shape[0]+key].values	
		out.append(dfs)
	return out


def convertToTimeLaggedSpace(s,E,tau):
	""" 
	Convert input to time lagged space using the embedded dimension, E,
	and time step, tau.
	
	Parameters
	----------
	s : pandas.core.series.Series or list of pandas.core.series.Series
		Input signal(s) to convert to time-lagged space.  If multiple signals are provided, the signals are "fused" together.  
	E : int
		Dimensional parameter
	tau : int
		Time lag parameter
		
	Returns
	-------
	P : pandas.core.frame.DataFrame
		Dataframe containing the input signal(s) converted to time-lagged space.  
	
	Example
	-------
	Example 1::
		
		s=_pd.Series(_np.arange(0,100))
		P=convertToTimeLaggedSpace(s, E=5, tau=1)
		
	Example 2::
		
		s1=_pd.Series(_np.arange(0,100))
		s2=_pd.Series(_np.arange(1000,1100))
		P12=convertToTimeLaggedSpace([s1,s2], E=5, tau=1)
		
	Example 3::
		
		s=_pd.Series(_np.arange(0,100))
		P=convertToTimeLaggedSpace(s, E=5, tau=5)
		
	Example 4::
		
		N=1000
		x,y,z=solveLorentz(N=N,plot='all')
		s=[pd.Series(x),pd.Series(z)]
		
		P=convertToTimeLaggedSpace(s, E=3, tau=2)
			
	"""
	if type(s) != list:
		s=[s]
	if type(s[0]) != _pd.core.series.Series:
		raise Exception('Input data should be a pandas Series or list of pandas Series')
	
	# initialize dataframe
	index=s[0].index.values[(E-1)*tau:]
	columns=_np.zeros(0,dtype=str)
	for i,si in enumerate(s):
		c=_np.arange(-(E-1)*tau,1,tau)
		c=['%d_%d'%(i+1,j) for j in c]
		columns=_np.concatenate((columns,c))
	P=_pd.DataFrame(index=index,columns=columns,dtype=float)	
	
	# populate dataframe, one Series and one column at a time
	for i,si in enumerate(s):
		c=_np.arange(-(E-1)*tau,1,tau)
		for j,cj in enumerate(c):
			name='%d_%d'%(i+1,cj)
			P[name]=si.iloc[cj+(E-1)*tau:si.shape[0]+cj].values	
		
	return P



# def forecast_2(s,T,tau):
# 	""" 
# 	Convert input to state space using the embedded dimension, E,
# 	and time step, tau 
# 	
# 	Example
# 	-------
# 	Example 1::
# 		
# 		s=_pd.Series(_np.arange(0,100))
# # 		P=convertToStateSpace(s, E=5, tau=1)[0]
# 		
# 	Example 2::
# 		
# 		s1=_pd.Series(_np.arange(0,100))
# 		s2=_pd.Series(_np.arange(1000,1100))
# # 		P1,P2=convertToStateSpace([s1,s2], E=5, tau=1)
# 		
# 	Example 3::
# 		
# 		s=_pd.Series(_np.arange(0,100))
# # 		P=convertToStateSpace(s, E=5, tau=5)[0]
# 		
# 	Example 4::
# 		
# 		
# 		
# # 		#s=createTentMap(300,plot=False)

# 		N=1000
# 		x,y,z=solveLorentz(N=N,plot='all')
# 		s=pd.Series(x)
# 		
# # 		P=convertToStateSpace(s, E=2, tau=1)[0]
# # 		
# # 		fig,ax=plt.subplots()
# # 		ax.plot(s,marker='.')
# # 		
# # 		fig,ax=plt.subplots()
# # 		ax.plot(P.iloc[:,0],P.iloc[:,1],'.',markersize=2)
# # 		ax.set_xlabel('Dimension 1')
# # 		ax.set_ylabel('Dimension 2')
# 		
# 	"""
# 	if type(s) != list:
# 		s=[s]
# 	out=[]
# 	for si in s:
# 		print('asdf')
# 		index=si.index.values[:-(T-0)*tau]
# 		columns=_np.arange(0,T+1,tau)#(-(E-1)*tau,1,tau)
# 		dfs=_pd.DataFrame(index=index,columns=columns)	
# 		for key,val in dfs.iteritems():
# 			print(key)
# 			dfs[key]=si.loc[index+key].values	
# # 			if key==0:
# # 				dfs[key]=si.iloc[key-columns.min():si.shape[0]+key].values	
# # 			else:
# # 				dfs[key]=si.iloc[key-columns.min():si.shape[0]+key].values	
# 		out.append(dfs)
# 	return out


def calcWeights(radii,method='exponential'):
	""" 
	Weights to be applied nearest neighbors 
	
	Example
	-------
	Example 1::
		
		# create data
		x=_np.arange(0,10+1)
		y=_np.arange(100,110+1)
		X,Y=_np.meshgrid(x,y)
		X=X.reshape((-1,1))
		Y=Y.reshape((-1,1))
		A=_np.concatenate((X,Y),axis=1)
		
		# points to investigate
		B=[[5.1,105.1],[8.9,102.55],[3.501,107.501]]
		
		numberOfNearestPoints=5
		points,indices,radii=findNearestNeighbors(A,B,numberOfNearestPoints=numberOfNearestPoints)
		
		weights=calcWeights(radii)
		
		for i in range(len(B)):
			fig,ax=_plt.subplots()
			ax.plot(X.reshape(-1),Y.reshape(-1),'.',label='original data')
			ax.plot(B[i][0],B[i][1],'x',label='point of interest')
			ax.plot(points[i][:,0],points[i][:,1],label='%d nearest neighbors\nwith weights shown'%numberOfNearestPoints,marker='o',linestyle='', markerfacecolor="None")
			plt.legend()
			
			w=weights.iloc[0]
			for j,v in weights.iloc[i].iteritems():
				ax.text(points[i][j,0],points[i][j,1],'%.3f'%v)
			
	
	"""
	if type(radii) in [_np.ndarray]:
		radii=_pd.DataFrame(radii)
	
	if method =='exponential':
		weights=_np.exp(-radii/radii.min(axis=1)[:,_np.newaxis])
		weights/=weights.sum(axis=1)[:,_np.newaxis] 	 
	elif method =='uniform':
		weights=_np.ones(radii.shape)/radii.shape[1]
	else:
		raise Exception('Incorrect weighting method provided')
	
	weights=_pd.DataFrame(weights,index=radii.index)
	return weights





def calcCorrelationCoefficient(data,fit,plot=False):
	""" 
	Pearson correlation coefficient 
	
	Reference
	---------
	Eq. 22 in https://mathworld.wolfram.com/CorrelationCoefficient.html
	
	Examples
	--------
	Example 1::
		
		## Test for positive correlation.  Simple.
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		y1=np.sin(2*np.pi*f*t)
		y2=y1+(np.random.rand(len(t))-0.5)*0.1
		calcCorrelationCoefficient(y1,y2,plot=True)
		
	Example 2::
		
		## Test for two cases at once.  
		
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		y1=np.sin(2*np.pi*f*t)
		y2=y1+(np.random.rand(len(t))-0.5)*0.1
		f=5.2e3
		y3=np.sin(2*np.pi*f*t)
		y4=y3+(np.random.rand(len(t))-0.5)*1
		df1=pd.DataFrame()
		df2=pd.DataFrame()
		df1['y1']=y1
		df2['y2']=y2
		df1['y3']=y3
		df2['y4']=y4
		calcCorrelationCoefficient(df1,df2,plot=True)
		
	Example 3::
		
		## Test for negative correlation.  Opposite of Example 1.
		
		import numpy as np
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		y1=np.sin(2*np.pi*f*t)
		y2=-y1+(np.random.rand(len(t))-0.5)*0.1
		calcCorrelationCoefficient(y1,y2,plot=True)
	"""
	if type(data)==_pd.core.frame.Series or type(data)==_np.ndarray:
		data=_pd.DataFrame(data)
		fit=_pd.DataFrame(fit)
	elif type(data)==_pd.core.frame.DataFrame:
		pass
	else:
		raise Exception('Improper data type')
		
	rho=_np.zeros(data.shape[1])
	for i,(key,val) in enumerate(data.iteritems()):
		y=val.values
		f=fit.iloc[:,i].values
		
		SSxy=((f-f.mean())*(y-y.mean())).sum()
		SSxx=((f-f.mean())**2).sum()
		SSyy=((y-y.mean())**2).sum()
		rho[i]=SSxy/_np.sqrt(SSxx*SSyy) # r-squared value
# 		rho[i]=SSxy**2/(SSxx*SSyy) # r-squared value
		
# 		rho[i]=_np.sqrt(rho[i]) # pearson correlation is sqrt(r^2)=r
	
		if plot==True:
			fig,ax=_plt.subplots()
			ax.plot(y,label='Original data')
			ax.plot(f,label='Reconstructed data')
			ax.legend()
			ax.set_title('%s, Rho = %.3f'%(key,rho[i]))
			
	return rho


	
# @deprecated
# def correlationCoefficient(data,fit,plot=False):
# 	#TODO retire this function
# 	""" 
# 	Correlation coefficient 
# 	
# 	Reference
# 	---------
# 	https://mathworld.wolfram.com/CorrelationCoefficient.html
# 	"""
# 	if type(data)==_pd.core.frame.DataFrame or type(data)==_pd.core.frame.Series:
# 		y=data.values.reshape(-1)
# 		f=fit.values.reshape(-1)
# 	elif type(data)==_np.ndarray:
# 		y=data.reshape(-1)
# 		f=fit.reshape(-1)
# 	SSxy=((f-f.mean())*(y-y.mean())).sum()
# 	SSxx=((f-f.mean())**2).sum()
# 	SSyy=((y-y.mean())**2).sum()
# 	rho=SSxy**2/(SSxx*SSyy)
# 	
# 	if plot==True:
# 		fig,ax=_plt.subplots()
# 		ax.plot(data,label='Original data')
# 		ax.plot(fit,label='Reconstructed data')
# 		ax.legend()
# 		ax.set_title('Rho = %.3f'%rho)
# 	return rho


	

# def calcRho(dfTActual,dfTFit,E,T,plot=True):
# 	""" calculates correlation coefficient between Fit and Actual data """
# 	dfRho=_pd.DataFrame(index=range(1,T+1))
# 	for t in range(1,T+1):
# 		if plot=='all':
# 			plotFitVsActual(dfTFit[str(t)],dfTActual[str(t)])
# 		dfRho.at[t,E]=correlationCoefficient(dfTActual[str(t)],dfTFit[str(t)])
# 		
# 	if plot==True or plot=='all':
# 		plotRho(dfRho)
# 	return dfRho


def findNearestNeighbors(X,Y,numberOfNearestPoints=1):
	"""
	Find the nearest neighbors in X to each point in Y
	
	Examples
	--------
	Example 1::
		
		# create data
		x=_np.arange(0,10+1)
		y=_np.arange(100,110+1)
		X,Y=_np.meshgrid(x,y)
		X=X.reshape((-1,1))
		Y=Y.reshape((-1,1))
		A=_np.concatenate((X,Y),axis=1)
		
		# points to investigate
		B=[[5.1,105.1],[8.9,102.55],[2,107]]
		
		numberOfNearestPoints=5
		points,indices,radii=findNearestNeighbors(A,B,numberOfNearestPoints=numberOfNearestPoints)
		
		for i in range(len(B)):
			fig,ax=_plt.subplots()
			ax.plot(X.reshape(-1),Y.reshape(-1),'.',label='original data')
			ax.plot(B[i][0],B[i][1],'x',label='point of interest')
			ax.plot(points[i][:,0],points[i][:,1],label='%d nearest neighbors'%numberOfNearestPoints,marker='o',linestyle='', markerfacecolor="None")
			_plt.legend()
			
	"""
	
	from sklearn.neighbors import NearestNeighbors

	neigh = NearestNeighbors(n_neighbors=numberOfNearestPoints)
	neigh.fit(X)
	radii,indices=neigh.kneighbors(Y)
	points=X[indices]
	
	return points, indices, radii


###################################################################################
#%% Main functions

def forecast(s,E,T,tau=1,knn=None,plot=False,weightingMethod=None,showTime=False):
	
	"""
	Create a map of s[first half] to s[second half] and forecast up to T steps into the future.
	
	Examples
	--------
	Example 1::
	
		N=1000
		s=createTentMap(N=N)
		E=3
		T=10
		tau=1
		knn=E+1
		
		forecast(s,E,T,tau,knn,True)
	"""
	#TODO start at T=0 instead of T=1

	N=s.shape[0]
	
	if knn == None or knn=='simplex':
		knn=E+1
	elif knn == 'smap':
# 	elif method == 'smap':
		knn=s.shape[0]//2-E+1
		
	if weightingMethod==None:
		weightingMethod='exponential'
	
	print("N=%d, E=%d, T=%d, tau=%d, knn=%d, weighting=%s"%(N,E,T,tau,knn,weightingMethod))
	
	start = _time.time()
	
	if showTime: printTime('Step 1 - Test and training data sets',start)
# 	if type(sy)==type(None):
	sX,sY,s=splitData(s)
# 	else:
# 		sX=sx.copy()
# 		sY=sy.copy()
		
	if showTime: printTime('Step 2 - Convert to state space',start)
	dfX,dfY=convertToStateSpace([sX,sY],E,tau)
# 	dfY=convertToStateSpace(sY,E,tau)
		
# 	if showTime: printTime('Step 3 - Find nearest neighbors',start)
# 	coordinates, indices, radii=findNearestNeighbors(dfX.values,dfY.values,knn)
# 	keysOfNearestNeighbors=_pd.DataFrame(indices+dfX.index[0],index=dfY.index)
# 	radii=_pd.DataFrame(radii,index=dfY.index)
		
	if showTime: printTime('step 3&4 - Create map and weights',start)
# 	if weightingFunction=='default':
# 		weightingFunction='exponential'
	keys,weights=createMap(dfX,dfY,knn,weightingMethod=weightingMethod)
		
	# reconstruct
# 	s_recon=reconstruct(s,keys,weights)
		
	if showTime: printTime('step 6 - Forecasting',start)

	dfFutureForecast,dfFutureActual=applyForecast(s,dfY,keys,weights,T)
# 	dfFutureForecast,dfFutureActual=forecast_2([s_recon,s.loc[s_recon.index]],T,tau)
# 	s_forecast=convertToStateSpace(s_recon,T,tau)[0]
	
# 	dfTActual,dfTGuess=forecast(sx,dfY,keys,weights,T=T)
				
	if showTime: printTime('step 6 - Calculate correlation coefficient, rho',start)
	dfRho=_pd.DataFrame(calcCorrelationCoefficient(dfFutureActual,dfFutureForecast,plot=False),index=dfFutureActual.columns.values)

	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(dfRho)
		_finalizeSubplot(	ax,
							xlabel='Steps into the future',
							ylabel='Correlation',
							ylim=[-0.01,1.01],
							xlim=[0,dfRho.index.max()+1],
							legendOn=False,
							title="N=%d, E=%d, T=%d, tau=%d, knn=%d, weighting=%s"%(N,E,T,tau,knn,weightingMethod))
		
# 		dfRho.plot()
	
	if showTime: printTime('done!',start)
	
	dic=dict({ 	'dfX':dfX,
				'dfY':dfY,
				'dfRho':dfRho,
				'dfFutureForecast':dfFutureForecast,
				'dfFutureActual':dfFutureActual,
				'keys':keys,
# 				'radii':radii,
				'weights':weights})
		
	return dic



# def SMIReconstruction_depricated(	s1A,
# 						s2A,
# 						s1B,
# 						E,
# 						tau,
# 						knn=None,
# 						s2B=None,
# 						plot=False,
# 						s1Name='s1',
# 						s2Name='s2',
# 						A='A',
# 						B='B',
# 						printStepInfo=False):
# 	"""
# 	
# 	Parameters
# 	----------
# 	s1A : pandas series
# 		Signal 1 (from source 1) and first half (e.g. A)
# 	s2A : pandas series
# 		Signal 2 (from source 2) and first half (e.g. A)
# 	s1B : pandas series
# 		Signal 1 (from source 1) and second half (e.g. B)
# 	s2B : pandas series
# 		Signal 2 (from source 2) and second half (e.g. B)
# 		Optional signal.  Is used to calculate rho if provided.  
# 	tau : int
# 		time step parameter
# 	knn : int
# 		number of nearest neighbors.  None is default = E+1
# 	plot : bool
# 		(Optional) plot
# 	
# 	
# 	Returns
# 	-------
# 	sB2_recon : pandas series
# 		Reconstructed sB2 signal
# 	rho : float
# 		Correlation value between sB2 and sB2_reconstruction.  Value is between 0 and where 1 is perfect agreement.  


# 	Notes
# 	-----
# 	  * This algorithm is based on https://doi.org/10.1088%2F1361-6595%2Fab0b1f
# 	
# 	
# 	Examples
# 	--------
# 	Example 1::
# 		
# 		import pandas as pd
# 		import matplotlib.pyplot as plt; plt.close('all')
# 		import johnspythonlibrary2 as jpl2
# 		
# 		N=10000
# 		T=1
# 		x,y,z=solveLorentz(N=N,T=T)
# 		t=np.linspace(0,T+T/N,N+1)
# 		s1A=pd.Series(x,index=t)
# 		s2A=pd.Series(z,index=t)
# 		
# 		x,y,z=solveLorentz(N=N,T=T,IC=[-9.38131377/2, -8.42655716/2 , 29.30738524/3])
# 		s1B=pd.Series(x,index=t)
# 		s2B=pd.Series(z,index=t)
# 		
# 		E=4
# 		knn=E+1
# 		tau=1
# 		
# 		sB2_recon,rho=SMIReconstruction(	s1A, 
# 											s2A, 
# 											s1B, 
# 											E, 
# 											tau, 
# 											s2B=s2B, 
# 											plot=True,
# 											s1Name='Lorentz-x',
# 											s2Name='Lorentz-z',
# 											A='IC1',
# 											B='IC2')
# 		fig=plt.gcf()
# 		ax=fig.get_axes()
# 		ax[0].set_xlim([0.2,0.25])
# 		ax[1].set_xlim([0.2,0.25])
# 		
# 	"""
# 	if type(s1A) != _pd.core.series.Series:
# 		raise Exception('sA1 should be in Pandas series format')
# 	if type(s2A) != _pd.core.series.Series:
# 		raise Exception('sA2 should be in Pandas series format')
# 	if type(s1B) != _pd.core.series.Series:
# 		raise Exception('sB1 should be in Pandas series format')
# 	if type(s2B) != None:
# 		if type(s2B) != _pd.core.series.Series:
# 			raise Exception('sB2 should be in Pandas series format')
# 			
# 	# index
# 	index_1=s1A.copy().index.values
# 	index_2=s2A.copy().index.values
# 	
# 	
# 	# reset the index to integers (this ensures all indices are the same and prevents issues with combining Series with possibly different indices in the future)
# 	s1A.index=_np.arange(0,s1A.shape[0],dtype=int)
# 	s2A.index=_np.arange(0,s2A.shape[0],dtype=int)
# 	s1B.index=_np.arange(0,s1B.shape[0],dtype=int)
# 	if type(s2B) != None:
# 		s2B.index=_np.arange(0,s2B.shape[0],dtype=int)
# 	
# 	if type(knn)==type(None):
# 		knn=E+1	# simplex method
# 		
# 	if printStepInfo==True:
# 		print("E = %d, \ttau = %d, \tknn = %d"%(E,tau,knn),end='')
# 		
# 	## remove offset
# 	s1A=s1A.copy()-s1A.mean()
# 	s2A=s2A.copy()-s2A.mean()
# 	s1B=s1B.copy()-s1B.mean()
# 	if type(s2B)!=type(None):
# 		s2B=s2B.copy()-s2B.mean()
# 		
# 	## convert to state space
# 	P1A,P2A,P1B=convertToStateSpace([s1A,s2A,s1B],E=E,tau=tau)

# 	## Create map from sA1 to sA2
# 	keys,weights=createMap(P1A,P1B,knn)
# 	
# 	## apply map to sB1 to get reconstructed sB2
# 	s2B_recon=reconstruct(s2A,keys,weights)
# 	
# 	## calc rho
# 	rho=calcCorrelationCoefficient(s2B[(E-1)*tau:],s2B_recon)	
# 	if printStepInfo==True:
# 		print(", \trho = %.3f"%rho)

# 	
# 	if plot==True and type(s2B)!=type(None):
# 		
# 		## sanity check map by reconstructing sA2 from sA1
# 		s1B_recon=reconstruct(s1A,keys,weights)
# 		rho_s1B=calcCorrelationCoefficient(s1B[(E-1)*tau:],s1B_recon)
# 		
# 		fig=_plt.figure()
# 		ax1 = _plt.subplot(221)
# 		ax2 = _plt.subplot(222, sharex = ax1)
# 		ax3 = _plt.subplot(223, sharex = ax1)
# 		ax4 = _plt.subplot(224, sharex = ax2)
# 		ax=[ax1,ax2,ax3,ax4]
# 		
# # 		fig,ax=_plt.subplots(2,2,sharex=False,sharey=True)
# 		ax[0].plot(index_1,s1A)
# 		_finalizeSubplot(ax[0],subtitle='%s %s'%(s1Name,A),legendOn=False,title='E = %d, tau = %d, knn = %d, N = %d'%(E, tau, knn,s1A.shape[0]),ylabel='Training data')
# 		print(index_1)
# 		ax[1].plot(index_1,s1B,label='original')
# 		ax[1].plot(index_1[ (s1A.shape[0]-s1B_recon.shape[0]):],s1B_recon,label='reconstructed')
# 		_finalizeSubplot(ax[1],subtitle='%s %s, rho=%.3f'%(s1Name,B,rho_s1B),legendLoc='lower right')
# 		ax[2].plot(index_2,s2A)
# 		_finalizeSubplot(ax[2],subtitle='%s %s'%(s2Name, A),legendOn=False,ylabel='Test data',xlabel='Time')
# 		ax[3].plot(index_2,s2B,label='original')
# 		ax[3].plot(index_2[(s1A.shape[0]-s2B_recon.shape[0]):],s2B_recon,label='reconstructed')
# 		_finalizeSubplot(ax[3],subtitle='%s %s, rho=%.3f'%(s2Name,B, rho),legendLoc='lower right',xlabel='Time')
# 		_finalizeFigure(fig,figSize=[6,4])
# 	
# 	## plot
# 	
# # 	ax=_plt.gca()
# # 	ax.set_title('results of applied map\nrho = %.3f\n%s'%(rho,ax.get_title()))
# 		
# 	return s2B_recon,rho



def SMIReconstruction(	s1A,
						s2A,
						s1B,
						E,
						tau,
						knn=None,
						s2B=None,
						plot=False,
						s1Name='s1',
						s2Name='s2',
						A='A',
						B='B',
						printStepInfo=False):
	"""
	
	Parameters
	----------
	s1A : pandas series
		Signal 1 (from source 1) and first half (e.g. A)
	s2A : pandas series
		Signal 2 (from source 2) and first half (e.g. A)
	s1B : pandas series
		Signal 1 (from source 1) and second half (e.g. B)
	s2B : pandas series
		Signal 2 (from source 2) and second half (e.g. B)
		Optional signal.  Is used to calculate rho if provided.  
	tau : int
		time step parameter
	knn : int
		number of nearest neighbors.  None is default = E+1
	plot : bool
		(Optional) plot
	
	
	Returns
	-------
	sB2_recon : pandas series
		Reconstructed sB2 signal
	rho : float
		Correlation value between sB2 and sB2_reconstruction.  Value is between 0 and where 1 is perfect agreement.  


	Notes
	-----
	  * This algorithm is based on https://doi.org/10.1088%2F1361-6595%2Fab0b1f
	
	
	Examples
	--------
	Example 1::
		
		import pandas as pd
		import matplotlib.pyplot as plt; plt.close('all')
		import johnspythonlibrary2 as jpl2
		import numpy as np
		
		N=10000
		T=1
		x,y,z=solveLorentz(N=N,T=T)
		t=np.linspace(0,T+T/N,N+1)
		s1A=pd.Series(x,index=t)
		s2A=pd.Series(z,index=t)
		
		x,y,z=solveLorentz(N=N,T=T,IC=[-9.38131377/2, -8.42655716/2 , 29.30738524/3])
		s1B=pd.Series(x,index=t)
		s2B=pd.Series(z,index=t)
		
		E=4
		knn=E+1
		tau=1
		
		sB2_recon,rho=SMIReconstruction(	s1A, 
											s2A, 
											s1B, 
											E, 
											tau, 
											s2B=s2B, 
											plot=True,
											s1Name='Lorentz-x',
											s2Name='Lorentz-z',
											A='IC1',
											B='IC2')
		
		fig=plt.gcf()
		ax=fig.get_axes()
		ax[0].set_xlim([0.2,0.25])
		ax[1].set_xlim([0.2,0.25])
		
		
	Example 2::
		
		## signal fusion case.  Use both x and y to reconstruct z
		
		import matplotlib.pyplot as plt; plt.close('all')
		
		N=2000
		x,y,z=solveLorentz(N=N-1)
		s1A=[_pd.Series(x[:N//2]),_pd.Series(y[:N//2])]
		s1B=[_pd.Series(x[N//2:]),_pd.Series(y[N//2:])]
		s2A=_pd.Series(z[:N//2])
		s2B=_pd.Series(z[N//2:])
		
		E=3
		tau=1

		sB2_recon,rho=SMIReconstruction(	s1A, 
									s2A, 
									s1B, 
									E, 
									tau, 
									s2B=s2B, 
									plot=True,
									s1Name='x and y fusion',
									s2Name='z',
									A='IC1',
									B='IC2')		
		
		sB2_recon,rho=SMIReconstruction(	_pd.Series(x[:N//2]), 
									s2A, 
									_pd.Series(x[N//2:]), 
									E, 
									tau, 
									s2B=s2B, 
									plot=True,
									s1Name='x only',
									s2Name='z',
									A='IC1',
									B='IC2')
		
		
		
		
	"""
	if type(s1A) != list:
		s1A = [s1A]
	if type(s1B) != list:
		s1B = [s1B]
		
	# make sure data is in pandas series format
	if type(s1A[0]) != _pd.core.series.Series:
		raise Exception('s1A should be in Pandas series format')
	if type(s2A) != _pd.core.series.Series:
		raise Exception('s2A should be in Pandas series format')
	if type(s1B[0]) != _pd.core.series.Series:
		raise Exception('s1B should be in Pandas series format')
	if type(s2B) != None:
		if type(s2B) != _pd.core.series.Series:
			raise Exception('s2B should be in Pandas series format')
			
	# make a copy of the index
	index_1=s1A[0].copy().index.values
	index_2=s2A.copy().index.values
	
	# reset the index to integers (this ensures all indices are the same and prevents issues with combining Series with possibly different indices in the future)
	for i in range(len(s1A)):
		s1A[i].index=_np.arange(0,s1A[i].shape[0],dtype=int)
		s1B[i].index=_np.arange(0,s1B[i].shape[0],dtype=int)
	s2A.index=_np.arange(0,s2A.shape[0],dtype=int)
	if type(s2B) != None:
		s2B.index=_np.arange(0,s2B.shape[0],dtype=int)
	
	# define number of nearest neighbors if not previously defined
	if type(knn)==type(None):
		knn=E+1	# simplex method
		
	if printStepInfo==True:
		print("E = %d, \ttau = %d, \tknn = %d"%(E,tau,knn),end='')
		
	## remove offset
	for i in range(len(s1A)):
		s1A[i]=s1A[i].copy()-s1A[i].mean()
		s1B[i]=s1B[i].copy()-s1B[i].mean()
	s2A=s2A.copy()-s2A.mean()
	if type(s2B)!=type(None):
		s2B=s2B.copy()-s2B.mean()
		
	## convert to time-lagged space
	P1A=convertToTimeLaggedSpace(s1A, E, tau)
	P1B=convertToTimeLaggedSpace(s1B, E, tau)

	## Create map from s1A to s1B
	keys,weights=createMap(P1A,P1B,knn)
	
	## apply map to s2A to get reconstructed s2B
	s2B_recon=reconstruct(s2A,keys,weights)
	
	## calc rho
	rho=calcCorrelationCoefficient(s2B[(E-1)*tau:],s2B_recon)	
	if printStepInfo==True:
		print(", \trho = %.3f"%rho)
	
	## optional plot
	if (plot==True or plot=='all') and type(s2B)!=type(None):
	
		## sanity check map by reconstructing sA2 from sA1
		if len(s1A)==1:
			s1B_recon=reconstruct(s1A[0],keys,weights)
			rho_s1B=calcCorrelationCoefficient(s1B[0][(E-1)*tau:],s1B_recon)

		
# 		## optional sanity checks 
# 		if plot=='all':
# 			keys,weights=createMap(P1A,P1A,knn)
# 			s1A_recon=reconstruct(s1A,keys,weights)
# 			s2A_recon=reconstruct(s2A,keys,weights)
# 			rho_s1A=calcCorrelationCoefficient(s1A[(E-1)*tau:],s1A_recon)
# 			rho_s2A=calcCorrelationCoefficient(s2A[(E-1)*tau:],s2A_recon)
		
		fig=_plt.figure()
		ax1 = _plt.subplot(221)
		ax2 = _plt.subplot(222, sharex = ax1)
		ax3 = _plt.subplot(223, sharex = ax1)
		ax4 = _plt.subplot(224, sharex = ax2)
		ax=[ax1,ax2,ax3,ax4]
		
		for i in range(len(s1A)):
			ax[0].plot(index_1,s1A[i],label='original_%d'%i)
# 		if plot=='all':
# 			ax[0].plot(index_1[(E-1)*tau:],s1A_recon,label='reconstructed')
# 			_finalizeSubplot(ax[0],subtitle='%s %s, rho=%.3f'%(s1Name,A,rho_s1A),legendOn=False,title='E = %d, tau = %d, knn = %d, N = %d'%(E, tau, knn,s1A.shape[0]),ylabel='Training data')
# 		if len(s1A)
		_finalizeSubplot(ax[0],subtitle='%s %s'%(s1Name,A),legendOn=(len(s1A)>1),title='E = %d, tau = %d, knn = %d, N = %d'%(E, tau, knn,s2A.shape[0]),ylabel='Training data')
			
		for i in range(len(s1B)):
			ax[1].plot(index_1,s1B[i],label='original_%d'%i)
		if len(s1A)==1:
			ax[1].plot(index_1[(E-1)*tau:],s1B_recon,label='reconstructed')
			_finalizeSubplot(ax[1],subtitle='%s %s, rho=%.3f'%(s1Name,B,rho_s1B),legendLoc='lower right')
		else:
			_finalizeSubplot(ax[1],subtitle='%s %s'%(s1Name,B),legendLoc='lower right')
		
		ax[2].plot(index_2,s2A,label='original')
# 		if plot=='all':
# 			ax[2].plot(index_2[(E-1)*tau:],s2A_recon,label='reconstructed')
# 			_finalizeSubplot(ax[2],subtitle='%s %s, rho=%.3f'%(s2Name,A,rho_s2A),legendLoc='lower right')
# 		else:
		_finalizeSubplot(ax[2],subtitle='%s %s'%(s2Name,A),legendLoc='lower right',legendOn=False)
		
		ax[3].plot(index_2,s2B,label='original')
		ax[3].plot(index_2[(E-1)*tau:],s2B_recon,label='reconstructed')
		_finalizeSubplot(ax[3],subtitle='%s %s, rho=%.3f'%(s2Name,B, rho),legendLoc='lower right',xlabel='Time')
		_finalizeFigure(fig,figSize=[6,4])
	
	return s2B_recon,rho
	


def ccm(	s1A,
			s1B,
			s2A,
			s2B,
			E,
			tau,
			knn=None,
			plot=False,
			removeOffset=False):
	
	"""
	
	Examples
	--------
	
	Example1::
		
		# lorentz equations
		N=1000
		x,y,z=solveLorentz(N=N-1,plot=False)
		
		# add noise
		x+=_np.random.normal(0,x.std()/1,x.shape[0])
		
		# prep data
		A=x[0:N//2]
		B=x[N//2:N]
		s1A=_pd.Series(A)
		s1B=_pd.Series(B)
		
		# call function
		rho=ccm(s1A,s1B,E=3,tau=1,plot=True)
		
			
	"""
	
	
	# make sure data is in pandas series format
	if type(s1A) != _pd.core.series.Series:
		raise Exception('s1A should be in Pandas series format')
	if type(s1B) != _pd.core.series.Series:
		raise Exception('s1B should be in Pandas series format')
	if type(s2A) != _pd.core.series.Series:
		raise Exception('s2A should be in Pandas series format')
	if type(s2B) != _pd.core.series.Series:
		raise Exception('s2B should be in Pandas series format')
	
	# define number of nearest neighbors if not previously defined
	if type(knn)==type(None):
		knn=E+1	# simplex method
		
	# reset the index to integers (this ensures all indices are the same and prevents issues with combining Series with possibly different indices in the future)
	s1A.index=_np.arange(0,s1A.shape[0],dtype=int)
	s1B.index=_np.arange(0,s1B.shape[0],dtype=int)
	s2A.index=_np.arange(0,s2A.shape[0],dtype=int)
	s2B.index=_np.arange(0,s2B.shape[0],dtype=int)
		
	# remove offset
	if removeOffset==True:
		s1A=s1A.copy()-s1A.mean()
		s1B=s1B.copy()-s1B.mean()
		s2A=s2A.copy()-s2A.mean()
		s2B=s2B.copy()-s2B.mean()
		
	# convert to time-lagged space
	P1A=convertToTimeLaggedSpace(s1A, E, tau)
	P1B=convertToTimeLaggedSpace(s1B, E, tau)	
	P2A=convertToTimeLaggedSpace(s2A, E, tau)
	P2B=convertToTimeLaggedSpace(s2B, E, tau)	
	
# 	fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
		
	## A to B
	keys,weights=createMap(P1A.copy(),P1B.copy(),knn)
	s2B_recon=reconstruct(s2A.copy(),keys,weights)
	rho_1to2=calcCorrelationCoefficient(s2B[(E-1)*tau:],s2B_recon,plot=False)	
	
 	## B to A
	keys,weights=createMap(P2A.copy(),P2B.copy(),knn)
	s1B_recon=reconstruct(s1A.copy(),keys,weights)
	rho_2to1=calcCorrelationCoefficient(s1B[(E-1)*tau:],s1B_recon,plot=False)	
	
	
	if plot==True:
		fig,ax=_plt.subplots(1,2,sharex=True,sharey=True)
		ax[1].plot(s1B[(E-1)*tau:],s1B_recon,linestyle='',marker='.')
		ax[0].plot(s2B[(E-1)*tau:],s2B_recon,linestyle='',marker='.')
		ax[0].set_aspect('equal')
		ax[1].set_aspect('equal')
		ax[0].plot([0,1],[0,1])  # TODO.  make this plot from min to max instead of 0 to 1
		ax[1].plot([0,1],[0,1])  # TODO.  make this plot from min to max instead of 0 to 1
		ax[0].set_title('s1 to s2 CCM')
		ax[1].set_title('s2 to s1 CCM')
		
	return rho_1to2, rho_2to1

###################################################################################
#%% exterior functions - functions that call the main functions


def eccm(s1,s2,E,tau,lagRange=_np.arange(-8,6.1,2),plot=False,s1Name='s1',s2Name='s2',title=''):
	
	print('work in progress.  not correct yet')
	N=s1.shape[0]
	
	results=_pd.DataFrame(index=lagRange)
# 	tau_d=0
	for lag in lagRange:
		print(lag)
# 		lag=-1
		
		if lag>0:
			s1_temp=s1[lag:].reset_index(drop=True)
			s2_temp=s2[:N-lag].reset_index(drop=True)
		elif lag<0:
 			s1_temp=s1[:lag]
 			s2_temp=s2[-lag:].reset_index(drop=True)
		else:
			s1_temp=s1.copy()
			s2_temp=s2.copy()
				 

		if False:
			fig,ax=_plt.subplots()
			ax.plot(s1)
			ax.plot(s2)
			ax.set_xlim([330,350])
			ax.set_title(lag)
			
		M=s1_temp.shape[0]
		s1A=s1_temp[:M//2].copy()
		s1B=s1_temp[M//2:].copy()
		s2A=s2_temp[:M//2].copy()
		s2B=s2_temp[M//2:].copy()
		
		if s1A.shape[0]>s1B.shape[0]:
			s1A=s1A[:-1]
			s2A=s2A[:-1]
		elif s1A.shape[0]<s1B.shape[0]:
			s1B=s1B[:-1]
			s2B=s2B[:-1]
		
			
# 		print(s1A.shape,s1B.shape,s2A.shape,s2B.shape)
		
		rho1,rho2=ccm(s1A,s1B,s2A,s2B,E,tau,plot=False)
# 		s1=s1
# 		rho_12=ccm(s1,s2,E=E,tau=tau,plot=False)
		results.at[-lag,'12']=rho1
# 		rho_21=ccm(s2,s1,E=E,tau=tau,plot=False)
		results.at[lag,'21']=rho2
		
	if plot==True:
		fig,ax=_plt.subplots()
		ax.plot(results['12'],label='%s to %s'%(s1Name,s2Name))
	# 	ax.plot(results.index.values*-1,results['12'])
		ax.plot(results['21'],label='%s to %s'%(s2Name,s1Name))
		ax.legend()
		ax.set_title(title)
		ax.set_ylabel('Pearson correlation')
		ax.set_xlabel('lag')
		
	return results
	
	
def example_Ye2015():
	
	import numpy as _np
	
	N=10000
# 	tau_d=0
	lagRange=(_np.arange(-8,8.1,1)).astype(int)
	E=2
	tau=1
# 	tau_d=0
		
	for tau_d in [0,2,4]:
		print(tau_d)
		
		s1,s2=twoSpeciesWithBidirectionalCausality(N=N,tau_d=tau_d,plot=False,params={'Ax':3.78,'Ay':3.77,'Bxy':0.07-0.00,'Byx':0.08-0.00})
	
		s1=s1[2000:3000].reset_index(drop=True)
		s2=s2[2000:3000].reset_index(drop=True)
		
		results=eccm(s1=s1,s2=s2,E=E,tau=tau,lagRange=lagRange,plot=True,title='tau_d = %d'%tau_d, s1Name='x',s2Name='y')
	

def ccmScan(s1,s2,E,tau,Nrange,plot=False,s1Name='s1',s2Name='s2'):
	print('work in progress.  needs an overhaul')
	"""

	Examples
	--------
	
	Example1::
		
# 		import matplotlib.pyplot as plt;plt.close('all')
		# Lorentz data
		Nmax=101000
		Narray=(10**_np.arange(1.8,_np.log10(Nmax-1000)+1e-4,0.025)).astype(int)
		x,y,z=solveLorentz(N=Nmax,plot=True,dt=0.05)#,IC=[-6.861764848645989/2, -0.5145288200456304 , 32.21788628404141])
		x=x[1000:]
		y=y[1000:]
		z=z[1000:]
		
# 		# add noise to Lorentz data
# 		x+=_np.random.normal(0,x.std()/15,x.shape[0])
# 		z+=_np.random.normal(0,z.std()/15,z.shape[0])
		
		# call function
		ccmScan(_pd.Series(x),_pd.Series(z),E=3,tau=2,Nrange=Narray,plot=True,s1Name='x',s2Name='z')

	"""
	
	results=_pd.DataFrame()
	for N in Nrange:
		print(N)
		
		s1A=_pd.Series(s1[0:N//2])
		s1B=_pd.Series(s1[N//2:N])
		s2A=_pd.Series(s2[0:N//2])
		s2B=_pd.Series(s2[N//2:N])
		
		
		rho_1to2,rho_2to1=ccm(s1A,s1B,s2A,s2B,E=E,tau=tau,plot=False)
		results.at[N,'rho12']=rho_1to2
		results.at[N,'rho21']=rho_2to1
# 		print(N,rho)
		
	if plot==True:
		fig,ax=_plt.subplots()
		ax.semilogx(results.rho12,label='%s to %s CCM'%(s1Name,s2Name))
		ax.semilogx(results.rho21,label='%s to %s CCM'%(s2Name,s1Name))
		ax.semilogx(Nrange,_np.ones(Nrange.shape),linestyle='--',color='grey')
		ax.set_xlabel('N')
		ax.set_ylabel('Pearson correlation')
		_finalizeSubplot(ax)

	return results



# def SMIParameterScan(sA1,sA2,sB1,ERange,tauRange,sB2=None,plot=False):
# 	print('work in progress')
# 	
# 	"""
# 	
# 	Examples
# 	--------
# 	Example 1 ::
# 		
# 		import pandas as pd
# 		import numpy as np
# 		
# 		N=2000
# 		dt=0.05
# 		x,y,z=solveLorentz(N=N,dt=dt)
# 		sA1=pd.Series(x)
# 		sB1=pd.Series(z)
# 		
# 		x,y,z=solveLorentz(N=N,dt=dt,IC=[-9.38131377/2, -8.42655716/2 , 29.30738524/3])
# 		sA2=pd.Series(x)
# 		sB2=pd.Series(z)
# 		
# 		ERange=np.arange(2,12+1,1)
# 		tauRange=np.arange(1,41+1)
# 		df=SMIParameterScan(sA1,sA2,sB1,ERange,tauRange,sB2=sB2,plot=True)
# 		
# 		ymax,xmax=_np.where(df.max().max()==df)
# 		tau_max=int(df.columns[xmax].values)
# 		E_max=int(df.index[ymax].values)
# 		
# 		SMIReconstruction(sA1, sA2, sB1, E_max, tau_max,sB2=sB2,plot=True)
# 		
# 	"""
# 	
# 	
# 	
# 	dfResults=_pd.DataFrame()
# 	for i,E in enumerate(ERange):
# 		for j,tau in enumerate(tauRange):
# 			print(E,tau)
# 	
# 			sB2_recon,rho=SMIReconstruction(sA1, sA2, sB1, E, tau,sB2=sB2,plot=False)
# 			dfResults.at[E,tau]=rho
# 			
# 	dfResults.index=ERange
# 	dfResults.columns=tauRange
# 	
# 	if plot==True:
# 		correlationHeatmap(tauRange,ERange,dfResults,xlabel='tau',ylabel='E')
# 		
# 	return dfResults


def SMIParameterScan2(s1A,s2A,s1B,ERange,tauRange,s2B=None,plot=False):
	print('work in progress')
	
	"""
	s1A : pandas series
		Signal 1 (from source 1) and first half (e.g. A)
	s2A : pandas series
		Signal 2 (from source 2) and first half (e.g. A)
	s1B : pandas series
		Signal 1 (from source 1) and second half (e.g. B)
	s2B : pandas series
		Signal 2 (from source 2) and second half (e.g. B)
		Optional signal.  Is used to calculate rho if provided.  
	tau : int
		time step parameter
	knn : int
		number of nearest neighbors.  None is default = E+1
	plot : bool
		(Optional) plot
	
	
	Examples
	--------
	Example 1 ::
		
		### note that this example cannot be run with "F9" in spyder.  Put it a script (if __name__=='__main__': etc) and run it with "F5" instead.
		
		if __name__ == '__main__':
			import pandas as pd
			import numpy as np
			
			N=10000
			dt=0.025
			
			# solve Lorentz equations with one set of ICs
			x,y,z=solveLorentz(N=N,dt=dt)
			s1A=pd.Series(x)
			s2A=pd.Series(z)
			
			# solve Lorentz equations with a second set of ICs
			x,y,z=solveLorentz(N=N,dt=dt,IC=[-9.38131377/2, -8.42655716/2 , 29.30738524/3])
			s1B=pd.Series(x)
			s2B=pd.Series(z)
			
			# perform reconstruction with a parameter scan of E and tau 
			ERange=np.arange(2,13+1,1)
			tauRange=np.arange(1,100+1)
			df=SMIParameterScan2(s1A=s1A,s2A=s2A,s1B=s1B, s2B=s2B,ERange=ERange,tauRange=tauRange,plot=True)
			
			fig=_plt.gcf()
			fig.savefig("SMIReconstruction_example_results.png",dpi=150)
			
			# plot best result
			ymax,xmax=_np.where(df.max().max()==df)
			tau_max=int(df.columns[xmax].values)
			E_max=int(df.index[ymax].values)
			SMIReconstruction(s1A=s1A,s2A=s2A,s1B=s1B, s2B=s2B, E=E_max, tau=tau_max,plot=True)
			
		
	"""
	# the pathos Pool function seems to be more compatible with "F5" operation than multiprocessing.Pool.  Neither can do "F9" operation.
	from pathos.multiprocessing import ProcessingPool as Pool
	from pathos.multiprocessing import cpu_count
# 	from multiprocessing import Pool,cpu_count

	# initialize pool
	pool = Pool(processes=cpu_count()-1) 
	pool.restart() # I don't know why I have to restart() the pool, but it often won't work without this command
	
	# wrapper function for multiprocessing
	def getResults(E,tau):
		print("E = %d, \ttau = %d"%(E,tau),end='')
		_,rho=SMIReconstruction(s1A=s1A,s2A=s2A,s1B=s1B,E=E,tau=tau,knn=None,s2B=s2B,plot=False)
		print(", \trho = %.3f"%rho)
		return E, tau, rho
		
	# launch pool
	X,Y=_np.meshgrid(ERange,tauRange) # generate each unique combination of E and tau
	results=pool.amap(getResults,X.reshape(-1),Y.reshape(-1))
	pool.close() # Indicate that no more data will be put on this queue by the current process. The background thread will quit once it has flushed all buffered data to the pipe
	pool.join() # wait until every process has finished
		
	# get results from pool
	results=results.get()
	
	# assign results to a dataframe
	dfResults=_pd.DataFrame()
	for result in results:
		E,tau,rho=result
		dfResults.at[E,tau]=rho
		
	if plot==True:
		print('Plotting results')
		correlationHeatmap(tauRange,ERange,dfResults,xlabel='tau',ylabel='E')
		
	print('Done!')
	return dfResults


def determineDimensionality(s,T,tau=1,Elist=_np.arange(1,10+1),method="simplex",weightingFunction='exponential',plot=False):
	"""
	
	Examples
	--------
	
	Example 1 ::
		
		N=2000
		s=createTentMap(N=N)
		T=10
		tau=1
		Elist=_np.arange(1,8+1)
		
		determineDimensionality(s,T,tau,Elist,plot=True)
	"""
	
	dfResults=_pd.DataFrame()
	for i,Ei in enumerate(Elist):
		knn=Ei+1
		dic=forecast(s,Ei,T,tau,knn,False)
		dfResults[Ei]=dic['dfRho'].iloc[:,0]
		
	dfResults=dfResults.transpose()

	if plot==True:
		dimensionHeatmap(dfResults,xlabel='T',ylabel='E')
		
		fig,ax=_plt.subplots()
		for i,(key,val) in enumerate(dfResults.iterrows()):
			print(key)
			ax.plot(val,label='E=%d'%key,marker='.')
		_finalizeSubplot(ax,xlabel='T',ylabel='rho')
		ax.set_title('N=%d, method=%s, tau=%d'%(len(s),method,tau))
	
	return dfResults


###################################################################################
#%% Examples

def example_sugihara1990():
	"""
	Forecasting example from Sugihara's 1990 paper.
	This function runs through a few cases and shows many of the intermediate
	steps in order to help better understand how the forecasting actually works.


	"""
	#TODO add a tau>1 case
	
	import matplotlib.pyplot as plt
	import johnspythonlibrary2 as jpl2
	import numpy as np

	N=100
	s=createTentMap(N=N)
	sx=s[:N//2]
	sy=s[N//2:]

	
	if True:
		# input signal data
		fig,ax=plt.subplots()
		ax.plot(sx,label='A',marker='x',markersize=4)
		ax.plot(sy,label='B',marker='+',markersize=4)
		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='Input signal, split in half (A and B)')
		fig.savefig('sugihara1990_figure1_inputdata.png')
		
	if True:
		# E=2 state space
		E=2
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		fig,ax=plt.subplots()
		ax.plot(Px[0],Px[-1],linestyle='',marker='x',label='A')
		ax.plot(Py[0],Py[-1],linestyle='',marker='+',markerfacecolor='none',label='B')
	
		jpl2.Plot.finalizeSubplot(	ax,
							xlabel='x(t)',
							ylabel='x(t-1)',
							title='E=2, time-lagged space')
		fig.savefig('sugihara1990_figure2_E2_timelagspace.png')
		
		
	if True:
		
		E=3
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		
		from mpl_toolkits.mplot3d import Axes3D
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(xs=Px[0].values.astype(np.float32),ys=Px[-1].values.astype(np.float32),zs=Px[-2].values.astype(np.float32),marker='x',label='A',linewidths=1.5)
		ax.scatter(xs=Py[0].values.astype(np.float32),ys=Py[-1].values.astype(np.float32),zs=Py[-2].values.astype(np.float32),marker='+',edgecolor='tab:blue',label='B',linewidths=2)
		ax.set_xlabel('x(t)')
		ax.set_ylabel('x(t-1)')
		ax.set_zlabel('x(t-2)')
		ax.view_init(elev=27., azim=18.)
		ax.legend()
		ax.set_title('E=3, time-lagged space')
		fig.savefig('sugihara1990_figure2_E3_timelagspace.png')
		
		
	E=2
	knn=E+1
	tau=1
	
	Px=convertToStateSpace(sx,E=E,tau=tau)[0]
	Py=convertToStateSpace(sy,E=E,tau=tau)[0]
	keys,weights=createMap(Px,Py,knn=knn)
	
	if True:
		
		fig,ax=plt.subplots()
		ax.plot(Px[0],Px[-1],linestyle='',marker='x',label='A')
		ax.plot(Py[0],Py[-1],linestyle='',marker='+',markerfacecolor='none',label='B')
		
		key=70 #55 52
		ax.plot(Py.loc[key][0],Py.loc[key][-1],'tab:red',marker='o',markersize=10,markeredgewidth=3,linewidth=2,label='Point in question',linestyle='',markerfacecolor='none')
		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			if i==0:
				label='E+1 neareast neighbors'
			ax.plot(Px.loc[k][0],Px.loc[k][-1],'tab:orange',marker='s',markersize=7,markeredgewidth=2,linewidth=2,label=label,markerfacecolor='none',linestyle='')
			
# 			Px[]
		jpl2.Plot.finalizeSubplot(	ax,
							xlabel='x(t)',
							ylabel='x(t-1)',
							title='E=2, time-lagged space.  Point in question with E+1 nearest neighbors.')
		fig.savefig('sugihara1990_figure3_nearestneighbors_timelagspace.png')
		
	
	if True:
		
		fig,ax=plt.subplots()
		ax.plot(sx,linestyle='-',marker='s',label='A',markersize=3)
		ax.plot(sy,linestyle='-',marker='o',label='B',markersize=3)
		
		key=70 #112 115
		ax.plot([key-1,key],[sy.loc[key-1],sy.loc[key]],'tab:red',marker='o',markersize=5,markeredgewidth=1.5,linewidth=1.5,label='Point in question',linestyle='-',markerfacecolor='none')
		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			if i==0:
				label='E+1 neareast neighbors'
			ax.plot([k-1,k],[sx.loc[k-1],sx.loc[k]],'tab:orange',marker='s',markersize=5,markeredgewidth=1.5,linewidth=1.5,label=label,markerfacecolor='none',linestyle='-')

		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='E=2, time-domain. Point in question with E+1 nearest neighbors.',
									xlim=[0,80])
									
		fig.savefig('sugihara1990_figure3_nearestneighbors_time_z.png')
		
	if True:
		dic_recon=forecast(s,E=E,T=10,plot=True)
		fig=plt.gcf()		
		fig.savefig('sugihara1990_figure4_forecast_analysis.png')
		
		
		
		fig,ax=plt.subplots()
		ax.plot(sx,linestyle='-',marker='s',label='A',markersize=3)
		ax.plot(sy,linestyle='-',marker='o',label='B',markersize=3)
		
		key=70 #112 115
		ax.plot([key-1,key],[sy.loc[key-1],sy.loc[key]],'tab:red',marker='o',markersize=5,markeredgewidth=1.5,linewidth=1.5,label='Point in question',linestyle='-',markerfacecolor='none')

		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			label2=''
			if i==0:
				label='E+1 neareast neighbors'
				label2='Data for forecast'
			ax.plot([k-1,k],[sx.loc[k-1],sx.loc[k]],'tab:orange',marker='s',markersize=5,markeredgewidth=1.5,linewidth=1.5,label=label,markerfacecolor='none',linestyle='-')
			ax.plot([k+1],[s.loc[k+1]],'tab:green',marker='s',markersize=5,markeredgewidth=2,linewidth=1.5,label=label2,markerfacecolor='none',linestyle='')

		dfFutureForecast=dic_recon['dfFutureActual']
		
		for i in [1]:#dfFutureForecast.columns.values:
			ax.plot(key+i,dfFutureForecast.at[key,i],linestyle='',marker='o',color='lime',markerfacecolor='none',markeredgewidth=2,markersize=5)

		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='Forecast 1 step.',
									xlim=[0,80])
									
		fig.savefig('sugihara1990_figure4_forecast_1.png')
		
		fig,ax=plt.subplots()
		ax.plot(sx,linestyle='-',marker='s',label='A',markersize=3)
		ax.plot(sy,linestyle='-',marker='o',label='B',markersize=3)
		
		key=70 #112 115
		ax.plot([key-1,key],[sy.loc[key-1],sy.loc[key]],'tab:red',marker='o',markersize=5,markeredgewidth=1.5,linewidth=1.5,label='Point in question',linestyle='-',markerfacecolor='none')

		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			label2=''
			if i==0:
				label='E+1 neareast neighbors'
				label2='Data for forecast'
			ax.plot([k-1,k],[sx.loc[k-1],sx.loc[k]],'tab:orange',marker='s',markersize=5,markeredgewidth=1.5,linewidth=1.5,label=label,markerfacecolor='none',linestyle='-')
# 			ax.plot([k+1],[s.loc[k+1]],'tab:green',marker='s',markersize=5,markeredgewidth=2,linewidth=1.5,label=label2,markerfacecolor='none',linestyle='')

		dfFutureForecast=dic_recon['dfFutureActual']
		
		for i in dfFutureForecast.columns.values:
			c='lime'
			label=''
			if i==0:
				c='tab:purple'
				label='Reconstruction'
			elif i==1:
				label='Forecast'
			ax.plot(key+i,dfFutureForecast.at[key,i],linestyle='',marker='o',color=c,markerfacecolor='none',markeredgewidth=2,markersize=5,label=label)
		
		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='Forecast 10 steps.',
									xlim=[0,80])
									
		fig.savefig('sugihara1990_figure4_forecast_10.png')
		
		
		
	if True:
		
		E=4
		knn=E+1
		tau=1
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		keys,weights=createMap(Px,Py,knn=knn)
		
		dic_recon=forecast(s,E=E,T=10,plot=True)
		
		fig,ax=plt.subplots()
		ax.plot(sx,linestyle='-',marker='s',label='A',markersize=3)
		ax.plot(sy,linestyle='-',marker='o',label='B',markersize=3)
		
		key=70 #112 115
		ax.plot(np.arange(key-E+1,key+1),sy.loc[key-E+1:key],'tab:red',marker='o',markersize=5,markeredgewidth=1.5,linewidth=1.5,label='Point in question',linestyle='-',markerfacecolor='none')

		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			label2=''
			if i==0:
				label='E+1 neareast neighbors'
				label2='Data for forecast'
			ax.plot(np.arange(k-E+1,k+1),sx.loc[k-E+1:k],'tab:orange',marker='s',markersize=5,markeredgewidth=1.5,linewidth=1.5,label=label,markerfacecolor='none',linestyle='-')
			ax.plot([k+1],[s.loc[k+1]],'tab:green',marker='s',markersize=5,markeredgewidth=2,linewidth=1.5,label=label2,markerfacecolor='none',linestyle='')

		dfFutureForecast=dic_recon['dfFutureActual']
		
		for i in [1]:#dfFutureForecast.columns.values:
			ax.plot(key+i,dfFutureForecast.at[key,i],linestyle='',marker='o',color='lime',markerfacecolor='none',markeredgewidth=2,markersize=5)

		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='E=4.  Forecast 1 step.',
									xlim=[0,80])
									
		fig.savefig('sugihara1990_figure5_forecast_1_E4.png')
		
		
	if True:
		
		
		N=1000
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]

		fig,ax=plt.subplots()
		
		ERange=np.arange(2,10)
			
		for E in ERange:
			knn=E+1
			tau=1
			
			Px=convertToStateSpace(sx,E=E,tau=tau)[0]
			Py=convertToStateSpace(sy,E=E,tau=tau)[0]
			keys,weights=createMap(Px,Py,knn=knn)
			
			dic_recon=forecast(s,E=E,T=10,plot=False)
			
			ax.plot(dic_recon['dfRho'],label='E=%d'%E,marker='.')
		
		jpl2.Plot.finalizeSubplot(ax,	
									xlabel='T (Forecast steps into the future)',
									ylabel='Correlation',
									title='Scan of E and forecast steps (T). \nN=%d, knn=%s, tau=%d'%(N,'E+1',tau))
	
		fig.savefig('sugihara1990_figure6_forecast_E_and_T_scan.png')



	if True:
		
		import matplotlib.pyplot as plt
			
		N=100
		s=createTentMap(N=N)
		sx=s[:N//2]
		sy=s[N//2:]
		
			
		E=3
		knn=E+1
		tau=2
		
		Px=convertToStateSpace(sx,E=E,tau=tau)[0]
		Py=convertToStateSpace(sy,E=E,tau=tau)[0]
		keys,weights=createMap(Px,Py,knn=knn)
		

		dic_recon=forecast(s,E=E,T=10,plot=False)
		fig=plt.gcf()		
# 		fig.savefig('sugihara1990_figure4_forecast_analysis.png')
		
		
		dfFutureForecast=dic_recon['dfFutureActual']
		
		fig,ax=plt.subplots()
		ax.plot(sx,linestyle='-',marker='s',label='A',markersize=3)
		ax.plot(sy,linestyle='-',marker='o',label='B',markersize=3)
		
		key=70 #112 115
		ax.plot([key-4,key-2,key],[sy.loc[key-4],sy.loc[key-2],sy.loc[key]],'tab:red',marker='o',markersize=5,markeredgewidth=1.5,linewidth=1.5,label='Point in question',linestyle='-',markerfacecolor='none')

		for i in range(knn):
			k=keys.loc[key,i]
			label=''
			label2=''
			if i==0:
				label='E+1 neareast neighbors'
				label2='Data for forecast'
			ax.plot([k-4,k-2,k],[sx.loc[k-4],sx.loc[k-2],sx.loc[k]],'tab:orange',marker='s',markersize=5,markeredgewidth=1.5,linewidth=1.5,label=label,markerfacecolor='none',linestyle='-')
			ax.plot([k+1],[s.loc[k+1]],'tab:green',marker='s',markersize=5,markeredgewidth=2,linewidth=1.5,label=label2,markerfacecolor='none',linestyle='')

		dfFutureForecast=dic_recon['dfFutureActual']
		
		for i in [1]:#dfFutureForecast.columns.values:
			ax.plot(key+i,dfFutureForecast.at[key,i],linestyle='',marker='o',color='lime',markerfacecolor='none',markeredgewidth=2,markersize=5)

		jpl2.Plot.finalizeSubplot(	ax,
									xlabel='Time',
									ylabel=r'$\Delta$x',
									title='E=3. Tau=2. Forecast 1 step.',
									xlim=[0,80])
									
		fig.savefig('sugihara1990_figure7_forecast_1_E3_tau2.png')
		
		
		
def example_lorentzAttractor_reconstruction():
	

	import matplotlib.pyplot as plt
	import numpy as np
	import johnspythonlibrary2 as jpl2
	
	N=2000-1
	x,y,z=solveLorentz(N=N,plot=True)
	
	N+=1
	x=x[N//2:]
	y=y[N//2:]
	z=z[N//2:]
	if True:
		fig=plt.gcf()
		fig.get_axes()[0].set_xlim(N//2,N)
		fig.savefig('lorentzReconstruction_figure1_time.png')
	
	if True:
		_,_,_=solveLorentz(N=N,plot='all')
		fig=plt.gcf()
		fig.savefig('lorentzReconstruction_figure1_statespace.png')

	N=N//2
	s1A=_pd.Series(x[:N//2])
	s1B=_pd.Series(x[N//2:])
	s2A=_pd.Series(z[:N//2])
	s2B=_pd.Series(z[N//2:])
	
	if True:
		fig,ax=plt.subplots(2,2)
		ax[0,0].plot(np.arange(0,N//2),s1A,label='Original')
		ax[0,1].plot(np.arange(N//2,N),s1B,label='Original')#,color='tab:blue')
		ax[1,0].plot(np.arange(0,N//2),s2A,label='Original')
		ax[1,1].plot(np.arange(N//2,N),s2B,label='Original')#,color='tab:blue')
		
		jpl2.Plot.finalizeSubplot(	ax[0,0],
									subtitle='x A (first half)',
									legendOn=False,
									xlim=[0,N//2-1],
									ylabel='x',
									title='A')
		jpl2.Plot.finalizeSubplot(	ax[0,1],
									subtitle='x B (second half)',
									legendOn=False,
									xlim=[N//2,N],
									title='B')
		jpl2.Plot.finalizeSubplot(	ax[1,0],
									subtitle='z A (first half)',
									legendOn=False,
									xlabel='Time',
									xlim=[0,N//2-1],
									ylabel='z',)
		jpl2.Plot.finalizeSubplot(	ax[1,1],
									subtitle='z B (second half)',
									legendOn=False,
									xlabel='Time',
									xlim=[N//2,N])
		ax[0,1].set_yticklabels([''])
		ax[1,1].set_yticklabels([''])
		ax[0,0].set_xticklabels([''])
		ax[0,1].set_xticklabels([''])
		fig.subplots_adjust(wspace=0.05,hspace=0.05)
		fig.suptitle('Take x and z signals and split in half (A and B)')
	
		fig.savefig('lorentzReconstruction_figure2_breakupData.png')
		
	E=3
	tau=1
	
	P1A=convertToTimeLaggedSpace(s1A, E=E, tau=tau)
	P1B=convertToTimeLaggedSpace(s1B, E=E, tau=tau)
	
	keys,weights = createMap(P1A,P1B,knn=E+1)
	
	s1B_recon=reconstruct(s1A,keys,weights)
	s1B_recon_rho=calcCorrelationCoefficient(s1B[E-1:],s1B_recon)
	if True:
		ax[0,1].plot(np.arange(N//2+E-1,N),s1B_recon,color='tab:blue',label='Reconstruction')
		jpl2.Plot.finalizeSubplot(	ax[0,1],
									subtitle='x B (second half). rho=%.3f'%s1B_recon_rho,
									legendOn=True,
									xlim=[N//2,N],
									title='B',
									legendLoc='lower right')
		fig.suptitle('For each point in x_B, use x_A to predict (reconstruct) x_B.\nThis provides a map from x_A to x_B with a near perfect reconstruction.\nE=%d, tau=%d, N=%d'%(E,tau,N//2))
		fig.savefig('lorentzReconstruction_figure3_createMap.png')
		
		
		
	
	s2B_recon=reconstruct(s2A,keys,weights)
	s2B_recon_rho=calcCorrelationCoefficient(s2B[E-1:],s2B_recon)
	if True:
		ax[1,1].plot(np.arange(N//2+E-1,N),s2B_recon,color='tab:blue',label='Reconstruction')
		jpl2.Plot.finalizeSubplot(	ax[1,1],
									subtitle='z B (second half). rho=%.3f'%s2B_recon_rho,
									legendOn=True,
									xlim=[N//2,N],
									xlabel='Time',
									legendLoc='lower right')
		fig.suptitle('Apply this same map to z_A to predict (reconstruct) z_B.\n The correlation for z_B_recon is almost as good as x_B_recon.\nE=%d, tau=%d, N=%d'%(E,tau,N//2))
		fig.savefig('lorentzReconstruction_figure4_applyMap2.png')
		
		
	if True:
		
		import pandas as pd
		import numpy as np
		
		N=3000
		dt=0.025
		
		# solve Lorentz equations with one set of ICs
		x,y,z=solveLorentz(N=N,dt=dt)
		s1A=pd.Series(x)
		s2A=pd.Series(z)
		
		# solve Lorentz equations with a second set of ICs
		x,y,z=solveLorentz(N=N,dt=dt,IC=[-9.38131377/2, -8.42655716/2 , 29.30738524/3])
		s1B=pd.Series(x)
		s2B=pd.Series(z)
		
		# perform reconstruction with a parameter scan of E and tau 
		ERange=np.arange(2,13+1,1)
		tauRange=np.arange(1,100+1)
		df=SMIParameterScan2(s1A=s1A,s2A=s2A,s1B=s1B, s2B=s2B,ERange=ERange,tauRange=tauRange,plot=True)
		fig=plt.gcf()
		ax=fig.get_axes()
		ax[0].set_title('Scan E and tau for an optimal solution.\nN=%d'%(N))
		fig.savefig('lorentzReconstruction_figure5_EAndTauScan.png')
		
		return df
		
	
	
def example_sugihara2012():
	
	"""
	Reconstruction of the figure's in Sugihara's 2012 paper on CCM
	"""
	
	print('work in progress.  the plots arent in amazing agreement for some reason')
	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	
		
		
	def function(N,rx=3.8,ry=3.5,Bxy=0.02,Byx=0.1, IC=[0.2,0.4],plot=False):
		x=np.zeros(N,dtype=float)
		y=np.zeros(N,dtype=float)
		x[0]=IC[0]
		y[0]=IC[1]
		
		for i in range(0,N-1):
			x[i+1]=x[i]*(rx-rx*x[i]-Bxy*y[i])
			y[i+1]=y[i]*(ry-ry*y[i]-Byx*x[i])
			
		if plot==True:
			fig,ax=plt.subplots()
			ax.plot(x)
			ax.plot(y)
		
		return x,y
	
	
	## Figure 3A
	if True:
		results=pd.DataFrame()
		for N in np.arange(20,3501,20).astype(int):
			
			x,y=function(N,rx=3.8,ry=3.5,Bxy=0.02,Byx=0.1,IC=[0.2,0.4],plot=False)
			
			s1A=pd.Series(x[:N//2])
			s1B=pd.Series(x[N//2:])
			s2A=pd.Series(y[:N//2])
			s2B=pd.Series(y[N//2:])
			
			E=2
			tau=1
			if N==1000:
				plot=False
			else:
				plot=False
			
			rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B,E=E,tau=tau,plot=plot)
		
			results.at[N,'rho_2B']=rho_2B
			results.at[N,'rho_1B']=rho_1B
			
			
		fig,ax=plt.subplots()
		ax.plot(results.rho_2B)
		ax.plot(results.rho_1B)
		
	## Figure 3C and 3D
	if True:
		N=1000
		
		x,y=function(N,rx=3.7,ry=3.7,Bxy=0.0,Byx=0.32,IC=[0.2,0.4],plot=False)
		
		s1A=pd.Series(x[:N//2])
		s1B=pd.Series(x[N//2:])
		s2A=pd.Series(y[:N//2])
		s2B=pd.Series(y[N//2:])
		
		E=2
		tau=1
		
		rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B,E=E,tau=tau,plot=True)
		
		
	## Figure 3B
	if True:
		
		E=2
		tau=1
		
		N=400
		results=_pd.DataFrame()
		
		for Bxy in np.arange(0,0.421,0.02):
			for Byx in np.arange(0,0.41,0.02):
				print(Bxy,Byx)
				x,y=function(N,rx=3.8,ry=3.5,Bxy=Bxy,Byx=Byx,IC=[0.2,0.4],plot=False)
								
				s1A=pd.Series(x[:N//2])
				s1B=pd.Series(x[N//2:])
				s2A=pd.Series(y[:N//2])
				s2B=pd.Series(y[N//2:])
				
				rho_1B, rho_2B = ccm(s1A,s1B,s2A,s2B,E=E,tau=tau,plot=False)
				results.at[Bxy,Byx]=rho_1B-rho_2B
		
		import xarray as xr
		da = xr.DataArray(results.values, 
				  dims=['Bxy', 'Byx'],
                  coords={'Bxy': np.arange(0,0.421,0.02),
				   'Byx': np.arange(0,0.41,0.02)})
		
		
		fig,ax=plt.subplots()
		from matplotlib import cm
		f=da.plot(levels=np.arange(-0.65,.66,.1),cmap=cm.Spectral_r,center=0,cbar_kwargs={'ticks': _np.arange(-1,1.01,0.1), 'label':'Rho-Rho'})
		fig.show()


# if __name__=='__main__':
#  	example_sugihara2012()
	