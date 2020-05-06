
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt

from deprecated import deprecated


def normalizeDF(df):
	""" Normalizes each column in a dataframe """
	return (df-df.mean(axis=0))/df.std(axis=0)


def filterDFByCol(df,string,invert=False):
	df=df.copy()
	index=df.columns.str.contains(string)
	
	if invert == True:
		index=_np.invert(index)
		
	return df.iloc[:,index]



def filterDFByIndex(df,string,invert=False):
	df=df.copy()
	index=df.index.str.contains(string)
		
	if invert == True:
		index=_np.invert(index)
		
	return df.iloc[index]


@deprecated(reason="Use either filterDFByIndex() or filterDFByCol() instead")
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
	
