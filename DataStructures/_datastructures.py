
import numpy as _np


# %% pandas related
## I find that a lot of pandas functionality is really difficult to access or remember.
## So I'm writing useful pandas functions here

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
