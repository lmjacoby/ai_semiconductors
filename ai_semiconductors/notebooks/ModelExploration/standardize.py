#!/usr/bin/env python
# coding: utf-8

# In[1]:


def standardize(v):
    import numpy as np
    """
    Takes a single column of a DataFrame and returns a new column 
    with the data standardized (mean 0, std deviation 1)
    """
    std = v.std()
    if std == 0:
        return np.zeros(len(v))
    else:
        return (v - v.mean()) / std

def normalize(v):
    """
    Takes a single column of a DataFrame and returns a new column 
    with the data normalized (data range[0,1])
    """
    max_ = v.max()
    min_ = v.min()
    if max_ == min_:
        return np.ones(len(v))
    else:
        return (v/(max_ - min_))