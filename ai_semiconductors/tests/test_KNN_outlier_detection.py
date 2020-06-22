#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")


def test_KNN_outlier_detection():
    import KNN_outlier_detection as KNN
    a = [[1, 2], [1, 2], [1, 2], [1, 2], [5, 4], [5, 4]]
    result = KNN.KNN_outlier_detection(0.01, a)
    assert len(result) == len(a), 'length of KNN output is wrong'


# In[ ]:
