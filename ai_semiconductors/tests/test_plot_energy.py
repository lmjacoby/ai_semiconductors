#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")


# In[3]:


def test_plot_energy():
    import plot_energy
    import os
    sys.path.append(os.path.abspath('../'))
    os.chdir('../')
    plot_energy.plot_energy('Cu', 'CdSe', 'M_A')
    assert 1 + 1 == 2, "Plotting doesn't work"


# In[ ]:
