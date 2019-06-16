import numpy as np

def entropy(arr):
    return -(arr*np.log2(arr+1e-10)
             + (1-arr)*np.log2(1-arr+1e-10))

def df2dict(df, key, value):
    return dict(zip(df[key].values, df[value].values))
