#!/usr/bin/env python
# coding: utf-8

# In[62]:


def KNN_outlier_detection(outlier_fraction, data):
    '''
    KNN/ABOD outlier removal
    '''

    from pyod.models.knn import KNN
    import numpy as np
    classifiers = {
         'K Nearest Neighbors (KNN)':  KNN(contamination=outlier_fraction),
         # 'Angle-based Outlier Detector (ABOD)':
         # ABOD(contamination=outlier_fraction)
    }
    input_matrix = np.array(data)
    y_pred = np.zeros((len(input_matrix), 1))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the dataset to the model
        clf.fit(input_matrix)

        # prediction of a datapoint category outlier or inlier
        y_pred[:, i] = clf.predict(input_matrix)

        # no of outliers in prediction
        n_outliers = int(y_pred[:, i].sum())
        print('No of Outliers : ', clf_name, n_outliers)
    return y_pred


# In[ ]:
