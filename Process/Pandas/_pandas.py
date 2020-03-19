
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt



def normalizeDF(df):
	return (df-df.mean())/df.std()


def filterDFByColOrIndex(df,string,col=True,invert=False):
	df=df.copy()
	if col==True:
		index=df.columns.str.contains(string)
	else:
		index=df.index.str.contains(string)
		
	if invert == True:
		index=_np.invert(index)
		
	if col==True:
		return df.iloc[:,index]
	else:
		return df.iloc[index]

	

def filterDFByTime(df,t1,t2):
	df=df.copy()
	return df[(df.index>=t1)&(df.index<=t2)]
	
