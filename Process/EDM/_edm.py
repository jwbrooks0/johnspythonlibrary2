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
# various generated signals to test the following code


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
	
	psoln = solve_ivp(	ODEs,
						[0,T],
						IC,  # initial conditions
						args=args,
						t_eval=_np.arange(0,T+dt,dt)
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
		ax[0].plot(x,marker='.');ax[0].set_ylabel('x')
		ax[0].set_title('Lorentz Attractor\n'+r'($\sigma$, b, r)='+'(%.3f, %.3f, %.3f)'%(args[0],args[1],args[2])+'\nIC = (%.3f, %.3f, %.3f)'%(IC[0],IC[1],IC[2]))
		ax[1].plot(y,marker='.');ax[1].set_ylabel('y')
		ax[2].plot(z,marker='.');ax[2].set_ylabel('z')
		
	if plot=='all':
		_plt.figure();_plt.plot(x,y)
		_plt.figure();_plt.plot(y,z)
		_plt.figure();_plt.plot(z,x)
	
		from mpl_toolkits.mplot3d import Axes3D
		fig = _plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(x,y,zs=z)
		
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
		vmin=0
		vmax=1
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
	index=dfY.index.values[:-T]
	dfTActual=_pd.DataFrame(index=index,dtype=float)
	for key in range(1,1+T):
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
	https://mathworld.wolfram.com/CorrelationCoefficient.html
	
	Examples
	--------
	Example 1::
		
		f=2e3
		t=np.arange(0,1e-3,2e-6)
		y1=np.sin(2*np.pi*f*t)
		y2=y1+(np.random.rand(len(t))-0.5)*0.1
		calcCorrelationCoefficient(y1,y2,plot=True)
		
	Example 2::
		
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
		rho[i]=SSxy**2/(SSxx*SSyy)
	
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
	

###################################################################################
#%% exterior functions - functions that call the main functions


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
# TODO this section needs to be overhauled
	

# def example5():
# 	""" work in progress """
# 	import xarray as xr
# 	import numpy as np
# 	Earray=np.arange(2,20)
# 	Tarray=np.arange(1,100)
# 	Tauarray=np.arange(1,2)
# 	da = xr.DataArray(
# 					  dims=['E', 'T','tau'],
# 	                  coords={'E': Earray,
# 					   'T': Tarray,
# 					   'tau': Tauarray,}
# 					   )
# 	def sigGen(t,phase=0):
# 		return _pd.Series(_np.sin(2*_np.pi*0.13e-1*t+phase))
# 	
# 	
# 	plot=False
# 	t=_np.arange(0,1001)
# 	syA=sigGen(t)
# 	sxA=syA*0.5+_np.random.uniform(-1,1,t.shape)*0.1
# #	sxA=sx[:sx.shape[0]//2]
# #	syA=sy[:sy.shape[0]//2]
# 	
# 	for i,E in enumerate(Earray):
# 		for j,T in enumerate(Tarray):
# 			for k,tau in enumerate(Tauarray):
# 				
# #	E=5
# #	T=10
# #	tau=1
# 				dic=main(sxA,E=E,T=T,sy=syA,tau=tau,method='simplex',plot=False,weightingFunction='default')
# 				dfTActual=dic['dfTActual']
# 				dfTGuess=dic['dfTGuess']
# 				if plot:
# 					fig,ax=_plt.subplots()
# 					ax.plot(sxA)
# 					ax.plot(dfTGuess['1'])
# 					
# 			
# 				syB=sigGen(t,_np.pi/2)
# 				sxB=syB*0.5+_np.random.uniform(-1,1,t.shape)*0.1
# 				
# 				dfY=convertToStateSpace(sxB,E=E,tau=tau)
# 				
# 				dfTActual,dfTGuess=forecast(sxB,dfY,dic['keysOfNearestNeighbors'],dic['weights'],T=1)
# 				if plot:
# 					fig,ax=_plt.subplots()
# 					ax.plot(sxB)
# 					ax.plot(dfTGuess)
# 				
# 				dfTGuess=dfTGuess.dropna()
# 				dfTActual=dfTActual.loc[dfTGuess.index]
# 				
# 				rho=correlationCoefficient(	dfTActual.values.reshape(-1),
# 									dfTGuess.values.reshape(-1))
# 	
# 				da.loc[E,T,tau]=rho
# 	
# 	
# def example6():
# # if True:
# 	""" frequency sweep with noise """
# 	# clean up.  maybe more work in the future
# 	
# 	import numpy as np
# # 	from scipy.signal import chirp
# 	t=np.arange(0,10000)
 	
# 	def genSig(t,f0,f1,phi=0,seed=1,tStep=1000):
# 		
# 		def chirp(t,fi,ff,phi):
# 			print(t[0],t[-1],fi,ff,phi)
# 			c=(ff-fi)/(t[-1]-t[0])
# 			phase=phi+2*np.pi*(c/2.0*(t-t[0])**2+fi*(t-t[0]))
# 			return np.sin(phase),phase[-1]
# 		
# 		y=np.zeros(t.shape)
# 		for i in range(len(t)//tStep):
# 			ti=t[i*tStep:(i+1)*tStep]
# 			if np.mod(i,2)==0:
# 				print(i)
# 				f_a=f0*1.0
# 				f_b=f1*1.0
# 			else:
# 				f_b=f0*1.0
# 				f_a=f1*1.0
# # 			yi=chirp(ti-ti[0],f_b,ti[-1]-ti[0],f_a,method='linear',phi=phi)
# 			yi,phi=chirp(ti,f_b,f_a,phi=phi)
# 			y[i*tStep:(i+1)*tStep]=yi
# 		np.random.seed(seed)
# 		noise=np.random.uniform(-1,1,len(t))*0.2
# 		return noise+y
# # 		return y
# 	
# 	y1=genSig(t,0.6e-2,0.1e-2,phi=0,seed=0)
# # 	plt.figure()
# # 	plt.plot(y1)
# 	

# 	y2=genSig(t,0.6e-2,0.1e-2,phi=90,seed=1)
# # 	y2=genSig(t,0.2e-1,0.2e-2,phi=90,seed=2)
# # 	y2=np.concatenate((y2,genSig(t,0.2e-1,0.2e-2,phi=90,seed=3)),axis=0)
# 	
# # 	t=np.arange(0,(t[-1]+1)*2)
# # 	np.random.seed(1)
# # 	noise=np.random.uniform(-1,1,len(t))*0.2
# # 	y2=noise+chirp(t,0.2e-1,t[-1],0.2e-2,method='linear',phi=90)
# # 	plt.plot(y)
# 		
# # 	y=noise
# # 	freqs=np.array([1.0/20,1.0/217])
# #	freqs=np.array([1.0/20])
# # 	for f in freqs:
# # 		y+=genSig(t,1,f)
# #	np.sin(2*np.pi*0.1*t)+np.sin(2*np.pi*0.007121*t)+noise
# 	sx1=_pd.Series(y1,index=t)
# 	sx2=_pd.Series(y2,index=t)
# 	
# 	_plt.close('all')
# 	fig,ax=_plt.subplots()
# 	_plt.plot(sx1)
# 	_plt.plot(sx2)
# 	
# #	determineDimensionality(sx,T=200,tau=1,Elist=np.arange(1,10+1))
# 	
# 	tau=1
# 	E=100
# 	T=1
# 	dic1=main(sx1,E=E,T=T,tau=tau)
# 	keysOfNearestNeighbors=dic1['keysOfNearestNeighbors']
# 	weights=dic1['weights']
# 	
# 	# forecasting	if showTime: printTime('Step 1 - Test and training data sets',start)

# 	sX2,sY2,sx2=splitData(sx2)
# 	
# 	dfY2=convertToStateSpace(sY2,E=E,tau=tau)
# 	dfTActual,dfTGuess=forecast(sx2,dfY2,keysOfNearestNeighbors,weights,T=T,plot=True)
# 			
# 	
# 	
# 	
# def example3():
# 	""" several sine waves with noise """
# 	# TODO work in progress
# 	
# 	import numpy as np
# 	np.random.seed(0)
# 	t=np.arange(0,1001)
# 	if True:
# 		noise=np.random.uniform(-1,1,len(t))*0.2
# 	else:
# 		noise=np.zeros(t.shape)
# 	
# 	def genSig(t,a,f,phi=0):
# 		return(a*np.sin(2*np.pi*f*t+phi))
# 		
# 	y=noise
# 	freqs=np.array([1.0/20,1.0/217])
# #	freqs=np.array([1.0/20])
# 	for f in freqs:
# 		y+=genSig(t,1,f)
# #	np.sin(2*np.pi*0.1*t)+np.sin(2*np.pi*0.007121*t)+noise
# 	sx=_pd.Series(y,index=t)
# 	
# 	_plt.close('all')
# 	fig,ax=_plt.subplots()
# 	_plt.plot(sx)
# 	
# #	determineDimensionality(sx,T=200,tau=1,Elist=np.arange(1,10+1))
# 	
# 	tau=1
# 	dic1=main(sx,E=1,T=200,tau=tau)
# 	dic3=main(sx,E=3,T=200,tau=tau)
# 	dic5=main(sx,E=5,T=200,tau=tau)
# 	dic10=main(sx,E=10,T=200,tau=tau)
# 	_plt.figure()
# 	_plt.plot(dic1['dfRho'])
# 	_plt.plot(dic3['dfRho'])
# 	_plt.plot(dic5['dfRho'])
# 	
# 	dfx=_pd.DataFrame()
# 	dfx['Time']=np.arange(1,sx.shape[0]+1)
# 	dfx['x']=sx
# 	import pyEDM
# #	pyEDM.EmbedDimension(	dataFrame=dfx,
# #							lib="0 500",
# #							pred="501 1000",
# #							Tp=200,
# #							maxE=100,
# #							columns='x',
# #							target='x')
# 	pyEDM.PredictInterval(	dataFrame=dfx,
# 							lib="0 500",
# 							pred="501 1000",
# 							E=10,
# 							maxTp=200,
# 							columns='x',
# 							target='x')
# 	fig=_plt.gcf()
# 	ax=fig.axes[0]
# 	ax.plot(dic10['dfRho'])
# 	

# 	
# def example4():
# 	""" 
# 	lorentz attractor
# 	"""
# 	#TODO work in progress
# 	#TODO the peaks in this final plot appear to be shifting to the left with increasing E and tau.  As best I can tell, pyEDM has the same issue.  What does this mean?
# 	
# 	def ODEs(t,y,*args):
# 		X,Y,Z = y
# 		sigma,b,r=args
# 		derivs=	[	sigma*(Y-X),
# 					-X*Z+r*X-Y,
# 					X*Y-b*Z]
# 		return derivs
# 	
# 	## solve
# 	from scipy.integrate import solve_ivp
# 	args=[10,8/3.,28] # sigma, b, r
# 	T=100
# 	dt=0.05 #0.05 max
# 	psoln = solve_ivp(	ODEs,
# 						[0,T],
# 						[-9.38131377, -8.42655716 , 29.30738524],
# #						[1,1,1],
# 						args=args,
# 						t_eval=_np.arange(0,T+dt,dt)
# 						)
# 	
# #	_plt.close('all')
# 	fig,ax=_plt.subplots(3,sharex=True)
# #	t=psoln.t
# 	x,y,z=psoln.y
# 	ax[0].plot(x,label='x')
# 	ax[1].plot(y,label='y')
# 	ax[2].plot(z,label='z')
# 	
# 	_plt.figure();_plt.plot(x,y)
# 	_plt.figure();_plt.plot(y,z)
# 	_plt.figure();_plt.plot(z,x)
# 	
# 	from mpl_toolkits.mplot3d import Axes3D
# 	fig = _plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.plot(x,y,zs=z)
# 	
# 	sx=_pd.Series(x,_np.arange(0,len(x)))
# 	sy=_pd.Series(y,_np.arange(0,len(x)))
# 	sz=_pd.Series(z,_np.arange(0,len(x)))
# 	
# 	T=500
# 	tau=5	
# 	N=sx.shape[0]
# 	dicAll=determineDimensionality(sx,T=T,tau=tau,Elist=_np.arange(1,10+1))
# 	fig=_plt.gcf()
# 	fig.savefig('N_%d_T_%d_tau_%d.png'%(N,T,tau))
# 	
# 	

# def example7(E=10,tau=1,knn='default',addNoise=False):
# 	""" 
# 	lorentz attractor SMI
# 	"""
# 	if knn=='default':
# 		knn=E+1 # simplex method
# 	#TODO work in progress
# 	#TODO the peaks in this final plot appear to be shifting to the left with increasing E and tau.  As best I can tell, pyEDM has the same issue.  What does this mean?
# 	
# 	_plt.close('all')
# 	
# 	
# 	x,y,z=solveLorentz([-9.38131377, -8.42655716 , 29.30738524],plot=True,addNoise=addNoise)
# 	x2,y2,z2=solveLorentz([-1, -8.42655716/2 , 29.30738524/2],plot=True,addNoise=addNoise)
# 	
# 	sx=_pd.Series(x-x.mean(),_np.arange(0,len(x)))
# 	sy=_pd.Series(y-y.mean(),_np.arange(0,len(x)))
# 	sz=_pd.Series(z-z.mean(),_np.arange(0,len(x)))
# 	
# 	sx2=_pd.Series(x2-x2.mean(),_np.arange(0,len(x2)))
# 	sy2=_pd.Series(y2-y2.mean(),_np.arange(0,len(x2)))
# 	sz2=_pd.Series(z2-z2.mean(),_np.arange(0,len(x2)))
# 	
# 	sA1=_pd.Series(x)
# 	sA2=_pd.Series(x2)
# 	sB1=_pd.Series(z)
# 	sB2=_pd.Series(z2)
# 	
# 			
# 	SMIReconstruction(sA1,sA2,sB1,E,tau,knn,sB2=sB2,plot=True)

# 	

# def example2():
# 	
# 	""" 
# 	Based loosely on Sugihara 1990 paper.
# 	This is intended to show some exmaples of how forecasting work.
# 	"""
# 	_plt.close('all')
# 	
# 	sx=createTentMap(100,plot=False)
# 	T=4
# 	E=3
# 	method="simplex"
# 	weightingFunction='exponential'
# 	dic=main(sx,E,T,method=method,weightingFunction=weightingFunction)
# 	dfY=dic['dfY']
# 	dfX=dic['dfX']
# 	weights=dic['weights']
# 	keysOfNearestNeighbors=dic['keysOfNearestNeighbors']
# 	fig,axAll=_plt.subplots(T,sharex=True)
# 	index=59
# 	for k,ax in enumerate(axAll):
# 		ax.plot(dfY.loc[index].index.values+index,dfY.loc[index],label='pattern to forecast',linewidth=5,color='tab:blue')
# 		for i,key in enumerate(keysOfNearestNeighbors.loc[index]):
# 			label1=''
# 			label2=''
# 			if i==0:
# 				label1='nearest neighbors'
# 				label2='Past similar points'
# 			ax.plot(dfX.loc[key].index.values+key,dfX.loc[key],color='tab:orange',label=label1,linewidth=5)
# 			ax.scatter(key+1+k,sx.loc[key+1+k],marker='o',color='orange',facecolor='none',label=label2,linewidths=2,s=50)
# 		
# 		ax.plot(sx.loc[:index],label='orig. data',color='k',marker='.',markersize=4)
# 		
# 		ypredict=[]
# 		x=[]
# 		for t in range(1,k+2):
# 		#	y=_pd.DataFrame(sx.loc[(keysOfNearestNeighbors.loc[index]+int(key)-1).values.reshape(-1)].values.reshape(shape),index=keysOfNearestNeighbors.index,columns=keysOfNearestNeighbors.columns)
# 		
# 			y=sx.loc[keysOfNearestNeighbors.loc[index].values+t].values
# 			w=weights.loc[index].values
# 			ypredict.append((y*w).sum())
# 			x.append(index+t)
# 			
# 		ax.scatter([index+t],[sx.loc[index+t]],marker='o',color='b',facecolor='none',label='next point to predict',s=50)
# 		ax.scatter(x[-1],ypredict[-1],marker='x',color='green',label='next predicted value')
# 	
# 		ax.plot(	_np.concatenate(([index],x)),
# 					_np.concatenate(([sx.loc[index]],ypredict)),
# 					linestyle='--',
# 					color='green',
# 					label='predicted',
# #						marker='x',
# 					ms=3
# 					)
# 		
# 		
# 			
# 		_finalizeSubplot(ax,
# 						  xlim=[0,index+5],
# 						  subtitle='T=%d'%(k+1),
# 						  legendOn=False)
# 		if k==0:
# 			_legendOutside(ax)
# 	axAll[0].set_title('E=%d, %s method, %s weighting'%(E,method,weightingFunction))
# 	ax.set_xlabel('Time')
# 	for i in range(2):
# 		_finalizeFigure(fig,h_pad=0.5,)


# def example1():
# 	""" 
# 	Sugihara 1990 paper.
# 	Effectively reproduces one of the figures in this paper and provides
# 	several supplementary plots
# 	"""
# 	
# 	
# 	## initialize
# 	_plt.close('all')
# 	sx=createTentMap(1000,plot=False)
# 	start = _time.time()
# 	printTime('Starting',start)
# 	
# 	# parameters		
# 	T=10						# steps into time to predict
# 	tau =1		
# 	Elist=_np.arange(1,10+1) 	# embedded dimension
# 	method="simplex"
# 	weightingFunction='exponential'
# 	
# 	# interate through each E
# 	for i,E in enumerate(Elist):
# 		
# 		# perform analysis and save rho
# 		dic=main(sx,E,T,tau,method=method,weightingFunction=weightingFunction)
# 		dfRho1=dic['dfRho']
# 		dfTGuess=dic['dfTGuess']
# 		dfTActual=dic['dfTActual']
# 		if i==0:
# 			dfRho=dfRho1
# 		else:
# 			dfRho=_pd.concat((dfRho,dfRho1),axis=1)
# 			
# 		# optional plots
# 		if E==3:
# 			fig,ax=_plt.subplots(2,2)
# 			ax[0][0].plot(sx,linewidth=0.4)
# 			_finalizeSubplot(ax[0][0],
# 									xlabel='Time (t)',
# 									ylabel=r'$\Delta x(t)$',
# 									legendOn=False)
# 			plotFitVsActual(dfTGuess['2'],dfTActual['2'],ax[0][1])
# 			plotFitVsActual(dfTGuess['5'],dfTActual['5'],ax[1][0])
# 			plotRho(_pd.DataFrame(dfRho[E]),ax[1][1])
# 			_finalizeFigure(fig,h_pad=3,w_pad=3,pad=1)
# #			fig.savefig('images/E3_figure.png')
# 			
# 			fig,ax=_plt.subplots()
# 			index=700
# 			ax.plot(sx,linewidth=5,color='grey',alpha=0.4,label='Original Data')
# 			ax.plot(dfTGuess.columns.values.astype(int)+index,
# 				   dfTGuess.loc[index+1],
# 				   color='tab:orange',
# 				   label='Forecast, t=%d'%index)
# 			_finalizeSubplot(ax,
# 							   xlabel='Time (t)',
# 							   ylabel=r'$\Delta x(t)$',
# 							   xlim=[index-4,index+4+10],
# 							   ylim=[-1,0.5],
# 							   title='Example forecast, E=%d'%E)
# #			fig.savefig('images/example_forecast.png')
# 		
# 		printTime('',start)
# 	
# 	# plot rho for each E
# 	fig,_=plotRho(dfRho,fig=fig)
# #	fig.savefig('images/results_summary.png')
# 	
# 	printTime('Done!',start)