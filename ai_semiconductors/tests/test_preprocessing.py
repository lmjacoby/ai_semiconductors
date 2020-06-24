#!/usr/bin/env python
# coding: utf-8

# In[17]:
import sys
sys.path.append("../")


def test_preprocessing():
    import preprocessing
    import os
    os.chdir('../')
    a, b, c, d, e, f = preprocessing.preprocessing('Cu', 'CdSe', 'M_A')
    assert not f, 'DFT check is right'


# In[ ]:
