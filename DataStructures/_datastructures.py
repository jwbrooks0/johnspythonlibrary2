
import numpy as _np
import os as _os
import importlib.util as _util
from pathlib import Path as _Path
# from collections import Sequence as _Sequence
from numpy.distutils.misc_util import is_sequence as _is_sequence


# %% pandas related

def filter_df_by_column(df, col_name, value, condition="=="):
    
    if condition == "==":
        df_out = df.loc[df[col_name] == value]
    elif condition == ">":
        df_out = df.loc[_np.greater(df[col_name], value)]
    elif condition == "<":
        df_out = df.loc[_np.less(df[col_name], value)]
    elif condition == ">=":
        df_out = df.loc[_np.greater_equal(df[col_name], value)]
    elif condition == "<=":
        df_out = df.loc[_np.less_equal(df[col_name], value)]
    elif condition == "!=":
        df_out = df.loc[_np.not_equal(df[col_name], value)]
    else:
        raise Exception(f"Condition not recognized: {condition}")
        
    return df_out


# %% xarray related

# TODO
