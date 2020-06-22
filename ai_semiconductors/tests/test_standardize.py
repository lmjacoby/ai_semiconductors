#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
sys.path.append("../")


# In[34]:


def test_standardize():
    import standardize
    import numpy as np
    a = np.array([1, 2, 3, 4, 5])
    std_a = (a - a.mean())/a.std()
    assert np.alltrue(std_a == standardize.standardize(a)),\
        "standardization is wrong"


# In[38]:


def test_normalize():
    import standardize
    import numpy as np
    a = np.array([1, 2, 3, 4, 5])
    nor_a = a / (a.max() - a.min())
    assert np.alltrue(nor_a == standardize.normalize(a)),\
        "normalization is wrong"


# In[ ]:
