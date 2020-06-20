#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../notebooks/ModelExploration")


# In[15]:


def test_MLR():
    from regression_models import MLR
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    tmse, rmse = MLR(X, y, X, y)
    assert type(rmse) == np.float64,\
        "output type for linear regression is wrong"


# In[26]:


def test_LassoReg():
    from regression_models import LassoReg
    from sklearn.model_selection._search import GridSearchCV as GCV
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3],
                  [1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = LassoReg(X, y)
    assert type(model) == GCV, "Lasso model construction is wrong"


# In[28]:


def test_Ridgereg():
    from regression_models import Ridgereg
    from sklearn.model_selection._search import GridSearchCV as GCV
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3],
                  [1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = Ridgereg(X, y)
    assert type(model) == GCV, "Ridge model construction is wrong"


# In[30]:


def test_Elastic():
    from regression_models import Elastic
    from sklearn.model_selection._search import GridSearchCV as GCV
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3],
                  [1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = Elastic(X, y)
    assert type(model) == GCV, "Elasticnet model construction is wrong"


# In[6]:


def test_krr():
    from regression_models import krr
    import pandas as pd
    import numpy as np
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3],
                  [1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    result = krr(X, y, X, y, ['test'])
    assert type(result) == pd.DataFrame,\
        "Output type of Kernel Ridge Regression model is wrong"


# In[ ]:
