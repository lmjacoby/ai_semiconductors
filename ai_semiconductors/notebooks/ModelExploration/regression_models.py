#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression


# In[3]:


def MLR(X_train, y_train, X_test, y_test):
    '''
    baseline: multivariate linear regression model
    '''
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))
    return rmse_train, rmse_test


# In[4]:


def LassoReg(X_train, y_train):
    '''
    Lasso regression, gridsearch on alpha values
    '''
    parameters = {'alpha': np.logspace(-7, 7, num=15)}
    reg = Lasso(random_state=0)
    clf = GridSearchCV(reg, parameters, cv=5, scoring="neg_mean_squared_error")
    clf.fit(X_train, y_train)
    return clf


# In[5]:


def Ridgereg(X_train, y_train):
    '''
    Ridge regression, gridsearch on alpha values
    '''
    parameters = {'alpha': np.logspace(-7, 7, num=15)}
    reg = Ridge(random_state=0)
    clf = GridSearchCV(reg, parameters, cv=5, scoring="neg_mean_squared_error")
    clf.fit(X_train, y_train)
    return clf


# In[ ]:


def Elastic(X_train, y_train):
    '''
    Elastic net model, gridsearch on alpha values and l1_ratio
    '''
    parameters = {'alpha': np.logspace(-7, 7, num=15),
                  'l1_ratio': np.linspace(0, 1, 10)}
    reg = ElasticNet(random_state=0)
    clf = GridSearchCV(reg, parameters, cv=5, scoring="neg_mean_squared_error")
    clf.fit(X_train, y_train)
    return clf


# In[6]:


def krr(X_train, y_train, X_test, y_test, predictors):
    '''
    Kernel ridge regression model,
    gridsearch on kernels and internal hyperparameters
    '''
    krr_result = []
    for i in range(len(predictors)):
        param_grid = [{"kernel": ['laplacian'], 'gamma': np.logspace(-5, 0, 6),
                       'alpha': [1e0, 1e-1, 1e-2, 1e-3]},
                      {"kernel": ['rbf'], 'gamma': np.logspace(-5, 0, 6),
                       'alpha': [1e0, 1e-1, 1e-2, 1e-3]},
                      {'kernel': ['poly'], 'degree':np.linspace(2, 6, 5),
                       'alpha': [1e0, 1e-1, 1e-2, 1e-3],
                       'gamma': np.logspace(-5, 0, 6)}]
        kr = GridSearchCV(KernelRidge(), param_grid=param_grid,
                          cv=5, scoring="neg_mean_squared_error")
        kr.fit(X_train, y_train)
        tmse = np.sqrt(mean_squared_error(y_train, kr.predict(X_train)))
        rmse = np.sqrt(mean_squared_error(y_test, kr.predict(X_test)))
        krr_result.append({
                            'Predictor': predictors[i],
                            'Training RMSE': tmse,
                            'Testing RMSE': rmse,
                            'Model': kr.best_estimator_,
                            'Best Parameter': kr.best_params_.values()
        })
    krr_result = pd.DataFrame(krr_result)
    return krr_result


# In[ ]:
