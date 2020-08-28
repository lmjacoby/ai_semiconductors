import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statistics import mean, stdev
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


def stratify_df(df, label_type, label_site):
    '''
    This function modifies the dataframe so that during cross validation
    the data can be split into test/train datasets that are equally stratified
    in "type" and "site" as the original dataframe.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - label_type: int. column index of "Type" column. Default: 1.
        - label_site int. column index of "Site" column. Dafault: 4.

    Outputs
        - b: pandas series. A newly encoded column that uniquely identifies
        the 15 possible combinations (3 sc types x 5 impurity sites) that a
        datapoint in the set could fal into.
    '''
    labels = df[df.columns[[label_type, label_site]]]

    # encode sc type and site columns, then combine them into a new string col
    # i.e. sctype 1 and site 3 becomes new column of 13 (dtype: string)
    enc = OrdinalEncoder(dtype=np.int)
    a = enc.fit_transform(labels)
    a = pd.DataFrame(a, columns=["SC_Type", "Site"])
    a = a.applymap(str)
    a = a[["SC_Type", "Site"]].apply(lambda x: ''.join(x), axis=1)

    # encode the new string col to 0-14 (15 total classes -
    # 3 sctypes x 5 defsites)
    b = np.array(a).reshape(-1, 1)
    b = enc.fit_transform(b)

    return b


def descriptors_outputs(df, d_start, o):
    '''
    This function splits to dataframe up into separate dataframes of
    descriptors and outputs by column.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - d_start: int. column index to that the descriptors columns start at.
        In the input df, the descriptors must start at some column at index
        df_start to the last column in the dataframe. Default: 3.
        - o: int. column index of the output. Deafult: 0.
    Outputs
        - X: pandas df. Dataframe with descriptors.
        - y: pandas df. Dataframe with output.
    '''
    X = df[df.columns[d_start:]]
    y = df[df.columns[o]]

    return X, y


def traintest(X, y, train_idx, test_idx):
    '''
    This function splits the descriptors (X) and output (y) points into train
    and test sets. The size of test set depends on the number of folds of CV.

    Inputs
        - X: pandas df. Dataframe with descriptors.
        - y: pandas df. Dataframe with output.
        - train_idx: np array. Indexes of training points.
        - test_idx: np array. Indexes of testing points.

    Outputs
        - X_train: np array. descriptor values of training data set.
        - X_test: np array. descriptor values of test data set.
        - y_train: np array. output values of training data set.
        - y_test: np array. output values of test data set.
    '''
    # train_idx and test_idx come from skf.split
    X_train, X_test = X.iloc[list(train_idx)], X.iloc[list(test_idx)]
    y_train, y_test = y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    return X_train, X_test, y_train, y_test


def fit_predict(X_train, y_train, X_test, clf):
    '''
    This function fits the training X/y data using the RFR model. Then makes a
    train and test prediction of the target value for each point, using the
    descriptors of training and testing. For each fold of the cross validation,
    the training and testing sets will change.

    Inputs
        - X_train: np array. descriptor values of training data set.
        - y_train: np array. output values of training data set.
        - X_test: np array. descriptor values of test data set.
        - clf: RandomForestRegressor from sklearn

    Outputs
        - trainpred: np array. predicted output value for every point in the
        train data set.
        - testpred: np array. predicted output value for every point in the
        test data set.
    '''
    # fit all the training data
    clf.fit(X_train, y_train)

    # predict on training data and testing data based on fit model
    trainpred = clf.predict(X_train)
    testpred = clf.predict(X_test)

    return trainpred, testpred


def rmse(y_train, y_test, trainpred, testpred):
    '''
    This function calculates the root mean squared error by evaluating the
    predicted values from the RFR model (clf) and the real values.

    Inputs
        - y_train: np array. output values of training data set.
        - y_test: np array. output values of test data set.
        - trainpred: np array. predicted output value for every point in the
        train data set.
        - testpred: np array. predicted output value for every point in the
        test data set.

    Outputs
        - train_rmse: float. root mean squared error of the predicted train
        points vs truth train points.
        - test_rmse: float. root mean squared error of the predicted test
        points vs truth test points.
    '''

    # compares the predicted points to the actual points and returns a RMSE val
    train_rmse = mean_squared_error(y_train, trainpred, squared=False)
    test_rmse = mean_squared_error(y_test, testpred, squared=False)

    return train_rmse, test_rmse


def rmse_list(train_rmse_list, test_rmse_list, train_rmse, test_rmse):
    '''
    This function appends the train/test rmse from each fold in the CV to
    a list.

    Inputs
        - train_rmse_list: list. empty list to append values to as rmse
        values for train data are calculated for CV fold.
        - test_rmse_list: list. empty list to append values to as rmse
        values for test data are calculated for CV fold.
        - train_rmse: float. root mean squared error of the predicted train
        points vs truth train points.
        - test_rmse: float. root mean squared error of the predicted test
        points vs truth test points.
    Outputs
        - train_rmse_list: list. list with appended rmse value for that fold
         of CV
        - test_rmse_list: list. list with appended rmse value for that fold
         of CV
    '''
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)

    return train_rmse_list, test_rmse_list


def rmse_total(df, X, y, train_idx, test_idx, clf, train_rmse_list,
               test_rmse_list):
    '''
    This is a wrapper func for the 'overall table'.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - X: pandas df. Dataframe with descriptors.
        - y: pandas df. Dataframe with output.
        - test_idx: np array. Array of indexes from CV.
        - train_idx: np array. Indexes of training points.
        - clf: RandomForestRegressor from sklearn
        - train_rmse_list: list. empty list to append values to as rmse
        values for train data are calculated for CV fold.
        - test_rmse_list: list. empty list to append values to as rmse
        values for test data are calculated for CV fold.

    Outputs
        - train_rmse_list: list. list of rmse values for train data. One RMSE
        value for each CV fold.
        - test_rmse_list: list. list of rmse values for test data. One RMSE
        value for each CV fold.
        - X_train: np array. descriptor values of training data set. returned
        from func to be used by later funcs.
        - y_train: np array. output values of training data set. returned
        from func to be used by later funcs.
    '''
    # this function is repeated for each fold of the CV

    # first the data is split into a train and test set based on the no of folds
    X_train, X_test, y_train, y_test = traintest(X, y, train_idx, test_idx)

    # then the model is fit on the test set and the train model makes
    # predictions for each point in the train and test set
    trainpred, testpred = fit_predict(X_train, y_train, X_test, clf)

    # those predictions are compared against the actual values to return
    # a RMSE for the train and test predictions
    train_rmse, test_rmse = rmse(y_train, y_test, trainpred, testpred)

    # then the RMSE is appended to a list (empty list established before the
    # repeated fold iteration) that will later be returned
    train_rmse_list, test_rmse_list = rmse_list(train_rmse_list,
                                                test_rmse_list,
                                                train_rmse, test_rmse)

    # return an updated RMSE list, with one new addition and the X and y train
    # points for a later function
    return train_rmse_list, test_rmse_list, X_train, y_train


def rmse_table_ms(train_rmse_list, test_rmse_list):
    '''
    This function adds a final row with the average rmse and std over the folds

    Inputs
        - train_rmse_list: list. list with appended rmse value for that fold
         of CV
        - test_rmse_list: list. list with appended rmse value for that fold
         of CV
    Outputs
        - rmse_df: pandas df. data frame with columns "train rmse"/"test rmse"
        with additional row of mean +/- standard deviation.
    '''
    # make a dataframe using the train and test rmse Lists. train/test are
    # separate columns

    d = {'train rmse': train_rmse_list,
         'test rmse': test_rmse_list}

    rmse_df = pd.DataFrame(data=d)

    # calculate the mean and standard deviation of the values in each column

    mean_train = rmse_df[rmse_df.columns[0]].mean()
    stddev_train = rmse_df[rmse_df.columns[0]].std()
    mean_test = rmse_df[rmse_df.columns[1]].mean()
    stddev_test = rmse_df[rmse_df.columns[1]].std()

    # append the mean and std dev as the last row in the dataframe

    rmse_df.loc[len(rmse_df)] = [str(round(mean_train, 2)) + ' +/- '
                                 + str(round(stddev_train, 3)),
                                 str(round(mean_test, 2)) + ' +/- '
                                 + str(round(stddev_test, 3))]

    return rmse_df


def df_tysi(df, train_idx, test_idx, output_type):
    '''
    This function divides each training and testing set up by type so the RFR
    model can predict on each type of sc separately. The model is trained
    on all the data.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - train_idx: np array. Indexes of training points.
        - test_idx: np array. Array of indexes from CV.
        - output_type: str. Value must be 'type' or 'site'.

    Outputs
        - train_idx_/test_idx_: list. indexes of II-VI, III-V, or IV-IV sc
         train/test points, or substitutional and interstitial defect sites.

    Notes: If output = 'type', returns 6 lists of indexes (train/test for each
    sc type). If output = 'site', returns 4 lists of indexes (train/test for
    each defect type (substitutional/interstitial))
    '''
    # output_type is designated as a parameter of the function. If it's 'type'
    # the points are sorted by semiconductor type (there are 3 types)

    if output_type == 'type':

        # make empty lists that will be filled with the indices of different
        # sc types

        train_idx_26 = []
        train_idx_35 = []
        train_idx_44 = []

        # train_idx is a list of the indices of the training point in the
        # original dataframe for each fold of CV. Each index is located in the
        # og df and the "Type" at that index is evaluated and then sorted into
        # on the of the three lists

        for idx in list(train_idx):
            if df['Type'].iloc[idx] == 'II-VI':
                train_idx_26.append(idx)
            elif df['Type'].iloc[idx] == 'III-V':
                train_idx_35.append(idx)
            elif df['Type'].iloc[idx] == 'IV-IV':
                train_idx_44.append(idx)

        # the same process happens for the test points

        test_idx_26 = []
        test_idx_35 = []
        test_idx_44 = []
        for idx in list(test_idx):
            if df['Type'].iloc[idx] == 'II-VI':
                test_idx_26.append(idx)
            elif df['Type'].iloc[idx] == 'III-V':
                test_idx_35.append(idx)
            elif df['Type'].iloc[idx] == 'IV-IV':
                test_idx_44.append(idx)

        # a separate index list of train and test points is returned for each
        # sc type (6 total)

        return (train_idx_26, train_idx_35, train_idx_44, test_idx_26,
                test_idx_35, test_idx_44)

    # the points are sorted by site, either substitutional or interstitial

    if output_type == 'site':

        train_idx_sub = []
        train_idx_int = []
        for idx in list(train_idx):

            # sub sites are made up of M_A and M_B

            if df['Site'].iloc[idx] == 'M_A' or df['Site'].iloc[idx] == 'M_B':
                train_idx_sub.append(idx)

            # int sites are made up of M_i_A and M_i_B and M_i_neut

            elif (df['Site'].iloc[idx] == 'M_i_A' or
                  df['Site'].iloc[idx] == 'M_i_B' or
                  df['Site'].iloc[idx] == 'M_i_neut'):
                train_idx_int.append(idx)

        test_idx_sub = []
        test_idx_int = []

        for idx in list(test_idx):
            if df['Site'].iloc[idx] == 'M_A' or df['Site'].iloc[idx] == 'M_B':
                test_idx_sub.append(idx)
            elif (df['Site'].iloc[idx] == 'M_i_A' or
                  df['Site'].iloc[idx] == 'M_i_B' or
                  df['Site'].iloc[idx] == 'M_i_neut'):
                test_idx_int.append(idx)

        # a separate index list of train and test points is returned for each
        # site (4 total)

        return (train_idx_sub, train_idx_int, test_idx_sub, test_idx_int)


def traintest_df_tysi(X, y, output_type, *args):
    '''
    This function takes the indexes of each type of sc or site of defect
    obtained in 'df_tysi'and creates an ordered dict where key/val pairs are
    one of four different dataframes (X_train, X_test, y_train, y_test), for
    each of the three types of semiconductors (12 pairs total) or each of
    the two types of defect sites (8 pairs total).

    Inputs
        - X: pandas df. Dataframe with descriptors.
        - y: pandas df. Dataframe with output.
        - output_type: str. Value must be 'type' or 'site'.
        - *args: arguements passed from df_tysi. 6 lists of indexs for 'type',
        4 lists of indexes for 'site'.
    Outputs
        - p: dict. A dictionary of key, val pairs. Values are 4 different
        dataframes (X_train, X_test, y_train, y_test), for each of the three
        types of semiconductors (12 dataframes total).
    '''
    if output_type == 'type':
        # num_list is for the key names

        num_list = [26, 35, 44]

        # args are the lists of train and test indexes by type. Each tuple is
        # the (train, test) of a different sc type.
        groups = [(args[0], args[3]), (args[1], args[4]),
                  (args[2], args[5])]

        p = OrderedDict()

        # make key/value pairs, and append to an ordered dictionary (this way,
        # the correct order can be passed on to the next function)

        for (num, grp) in zip(num_list, groups):
            p['X_train_{0}'.format(num)] = X.iloc[grp[0]]
            p['X_test_{0}'.format(num)] = X.iloc[grp[1]]
            p['y_train_{0}'.format(num)] = y.iloc[grp[0]]
            p['y_test_{0}'.format(num)] = y.iloc[grp[1]]

    if output_type == 'site':
        # site_list is for the key names
        site_list = ['sub', 'int']

        # args are the lists of train and test indexes by site. Each tuple is
        # the (train, test) of a different site.
        groups = [(args[0], args[2]), (args[1], args[3])]

        p = OrderedDict()

        for (site, grp) in zip(site_list, groups):
            p['X_train_{0}'.format(site)] = X.iloc[grp[0]]
            p['X_test_{0}'.format(site)] = X.iloc[grp[1]]
            p['y_train_{0}'.format(site)] = y.iloc[grp[0]]
            p['y_test_{0}'.format(site)] = y.iloc[grp[1]]

    return p


def make_xy_list(p, output_type):
    '''
    This function takes the values from the dict in traintest_df_tysi, and
    puts them into tuples of (X_train, X_test, y_train, y_test) by type or
    site so they can be used in the subsequent func, fit_predict_tysi.

    Inputs
        - p: dict. A dictionary of key, val pairs. Values are 4 different
        dataframes (X_train, X_test, y_train, y_test), for each of the three
        types of semiconductors (12 dataframes total).
        - output_type: str. Value must be 'type' or 'site'.
    Outputs
        - xytype_list: list of tuples. tuples of (X_train, X_test, y_train,
         y_test) by type (3) so they can be used in the subsequent func,
         type_fit_predict
    '''
    # make the values from the ordered dictionary into tuples by type or site.
    # type will have 3 tuples of 4 values each (X_train, X_test, y_train, y_test)

    if output_type == 'type':
        a = (tuple(p.values())[:4])
        b = (tuple(p.values())[4:8])
        c = (tuple(p.values())[8:12])

        # list of tuples to be used is fit_predict_tysi and wrapper_tysi
        xy_list = [a, b, c]

    # site will have 2 tuples of 4 values each (X_train, X_test, y_train, y_test)

    if output_type == 'site':
        a = (tuple(p.values())[:4])
        b = (tuple(p.values())[4:8])

        xy_list = [a, b]

    return xy_list


def fit_predict_tysi(clf, val, X_train, y_train):
    '''
    This function fits the model on all the data, and predicts each type of sc
    or sub/int defect site separately. To do this it uses the list of tuples
    from make_xy_list or make_xy_list.

    Inputs
        - clf: RandomForestRegressor from sklearn
        - val: tuple. tuple from xy_list
        - X_train: np array. descriptor values of training data set.
        - y_train: np array. output values of training data set.
    Outputs
        - trainpred: np array. predicted output value for every point in
        the train data set (for each type or site).
        - testpred: np array. predicted output value for every point in
        the test data set (for each type or site).
    '''
    # the model is trained using all the points (ie not descriminating by
    # type or site)
    clf.fit(X_train, y_train)

    # val is a tupple of (X_train, X_test, y_train, y_test) from make_xy_list
    trainpred = clf.predict(val[0])
    testpred = clf.predict(val[1])

    return trainpred, testpred


def rmse_tysi(val, trainpred, testpred):
    '''
    This function calculates the root mean squared error by evaluating the
    predicted values from the RFR model (clf) and the real values.

    Inputs
        - val: tuple. tuple from xytype_list or xysite_list
        - trainpred: np array. predicted output value for every point in
        the train data set (for each type or site).
        - testpred: np array. predicted output value for every point in
        the test data set (for each type or site).
    Outputs
        - train_rmse: float. root mean squared error of the predicted
         train points vs truth train points.
        - test_rmse: float. root mean squared error of the predicted test
        points vs truth test points.
    '''
    # val is a tupple of (X_train, X_test, y_train, y_test) from make_xy_list

    train_rmse = mean_squared_error(val[2], trainpred, squared=False)
    test_rmse = mean_squared_error(val[3], testpred, squared=False)

    return train_rmse, test_rmse


def make_dict_tysi(val, df, train_rmse, test_rmse,
                   train_dict, test_dict, output_type):
    '''
    Adds values to two Default dictionaries (type/sitetrain_dict and
    type/sitetest_dict) which will concatenate all values with the same
    key. Thus, the key is either 'train rmse + sctype' or 'test rmse +
    sc type' (in the case of SC type) or 'train rmse + sub/int' or
    'test rmse + sc sub/int' (in the case of site). This creates 3 (type) or
    2 (site) unique key, value pairs per dictionary.

    Inputs
        - val: tuple. tuple from xy_list
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - train_rmse: float. root mean squared error of the predicted
         train points vs truth train points.
        - test_rmse: float. root mean squared error of the predicted test
        points vs truth test points.
        - train_dict: default dict. A default dict to add key, val
        pairs to for each sc type of the training data, or sub/int defect site.
        - test_dict: default dict. A default dict to add key, val
        pairs to for each sc type of the testing data, or sub/int defect site.
        - output_type: str. Value must be 'type' or 'site'.

    Outputs
        - train_dict: default dict. A default dict to add key, val
        pairs to for each sc type of the training data, or sub/int defect site.
        - test_dict: default dict. A default dict to add key, val
        pairs to for each sc type of the testing data, or sub/int defect site.
    '''
    # this function it iterated through for all the tuples in xy_list

    # index is a list of the indeces of X_train for a particular sc type or
    # site. This func uses the first val in this list (though it could have
    # been any of the vals) to locate the proper 'Type' or 'Site' for the key

    index = val[0].index.tolist()

    # this is to name the key by type or site so that it can be associated with
    # the train and test rmse list
    if output_type == 'type':
        key_train = ('train rmse ' + str(df['Type'].loc[index[0]]))
        key_test = ('test rmse ' + str(df['Type'].loc[index[0]]))

    elif output_type == 'site':
        if (df['Site'].loc[index[0]] == 'M_A' or
                df['Site'].loc[index[0]] == 'M_B'):
            key_train = ('train rmse ' + 'sub')
            key_test = ('test rmse ' + 'sub')
        elif (df['Site'].loc[index[0]] == 'M_i_A' or
              df['Site'].loc[index[0]] == 'M_i_B' or
              df['Site'].loc[index[0]] == 'M_i_neut'):
            key_train = ('train rmse ' + 'int')
            key_test = ('test rmse ' + 'int')
        else:
            pass
    else:
        pass

    # append the appropriate train and test RMSE for each fold of CV
    # to the appropriate default dictionary.
    train_dict[key_train].append(train_rmse)
    test_dict[key_test].append(test_rmse)

    # the dictionaries with the updated RMSE after each round of CV are returned
    return train_dict, test_dict


def wrapper_tysi(df, clf, X_train, y_train, train_dict, test_dict,
                 train_idx, test_idx, X, y, output_type):
    '''
    wrapper func for the RMSE tables.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - clf: RandomForestRegressor from sklearn
        - X_train: np array. descriptor values of training data set.
        - y_train: np array. output values of training data set.
        - train_dict: default dict. A default dictionary to add key, val
        pairs to for each sc type of the training data
        - train_dict: default dict. A default dictionary to add key, val
        pairs to for each sc type of the testing data
        - train_idx: np array. Indexes of training points.
        - test_idx: np array. Array of indexes from CV.
        - X_train: np array. descriptor values of training data set.
        - y_train: np array. output values of training data set.
        - output_type: str. Value must be 'type' or 'site'.
    Outputs
        - train_dict: default dict. A complete default dict to add key, val
        pairs to for each sc type of the training data
        - train_dict: default dict. A complete default dict to add key, val
        pairs to for each sc type of the testing data
    '''
    # this func happens for each fold of CV
    if output_type == 'type':
        # divide train and test set up by sc type
        (train_idx_26, train_idx_35, train_idx_44, test_idx_26,
         test_idx_35, test_idx_44) = df_tysi(df, train_idx, test_idx,
                                             output_type)
        # make a dictionary (p) of 12 dataframes (4 for each sc type - X_train,
        # X_test, y_train, y_test)
        p = traintest_df_tysi(X, y, output_type, train_idx_26, train_idx_35,
                              train_idx_44, test_idx_26, test_idx_35,
                              test_idx_44)

    elif output_type == 'site':
        # divide train and test set up by site

        (train_idx_sub, train_idx_int, test_idx_sub,
         test_idx_int) = df_tysi(df, train_idx, test_idx,
                                 output_type)
        # make a dictionary (p) of 8 dataframes (4 for each site - X_train,
        # X_test, y_train, y_test)
        p = traintest_df_tysi(X, y, output_type, train_idx_sub, train_idx_int,
                              test_idx_sub, test_idx_int)

    # take p and make into tuples of (X_train, X_test, y_train, y_test)

    xy_list = make_xy_list(p, output_type)

    # iterate through each tuple in xy_list

    for xy in xy_list:

        # train model on all data, predict train and test sets for a particular
        # sc type OR site

        trainpred, testpred = fit_predict_tysi(clf, xy,
                                               X_train, y_train)

        # calculate the RMSE based on the results of fit_predict_tysi

        train_rmse, test_rmse = rmse_tysi(xy, trainpred, testpred)

        # add train and test RMSE to a default dictionary

        train_dict, test_dict = make_dict_tysi(xy, df, train_rmse, test_rmse,
                                               train_dict, test_dict,
                                               output_type)

    return train_dict, test_dict


def rmse_table_tysi(train_dict, test_dict, output_type):
    '''
    This func turns the default dictionaries into pandas df, and then pieces
    them together so the train rmse/test rmse for each type can be displayed
    in a table.

    Inputs
        - train_dict: default dict. A complete default dict to add key, val
        pairs to for each sc type of the training data
        - train_dict: default dict. A complete default dict to add key, val
        pairs to for each sc type of the testing data
        - output_type: str. Value must be 'type' or 'site'.

    Outputs
        - rmse_df_: pandas df. Table of train/test rmse values where
        each row is rmse for a fold in the cross validation.

    Notes: If output = 'type', returns 3 rmse tables (one for each sc type).
    If output = 'site', returns 2 rmse tables (one for each type of defect
    site (substitutional/interstitial)).
    '''
    # turn the train adn test default dictionaries into dfs. keys = columns.
    train_rmse_df = pd.DataFrame.from_dict(train_dict)
    test_rmse_df = pd.DataFrame.from_dict(test_dict)

    if output_type == 'type':

        # concatenate train and test for each sc type.

        rmse_df_26 = pd.concat([train_rmse_df['train rmse II-VI'],
                               test_rmse_df['test rmse II-VI']], axis=1)
        rmse_df_35 = pd.concat([train_rmse_df['train rmse III-V'],
                               test_rmse_df['test rmse III-V']], axis=1)
        rmse_df_44 = pd.concat([train_rmse_df['train rmse IV-IV'],
                               test_rmse_df['test rmse IV-IV']], axis=1)

        # make a list of the new dataframes
        df_list = [rmse_df_26, rmse_df_35, rmse_df_44]

    if output_type == 'site':
        # concatenate train and test for each site.

        rmse_df_sub = pd.concat([train_rmse_df['train rmse sub'],
                                test_rmse_df['test rmse sub']], axis=1)
        rmse_df_int = pd.concat([train_rmse_df['train rmse int'],
                                test_rmse_df['test rmse int']], axis=1)

        df_list = [rmse_df_sub, rmse_df_int]

    # append a line ot the end of each dataframe with the mean and std dev of the vals

    for dtfr in df_list:
        mean_train = dtfr[dtfr.columns[0]].mean()
        stddev_train = dtfr[dtfr.columns[0]].std()
        mean_test = dtfr[dtfr.columns[1]].mean()
        stddev_test = dtfr[dtfr.columns[1]].std()

        dtfr.loc[len(dtfr)] = [str(round(mean_train, 2)) + ' +/- '
                               + str(round(stddev_train, 3)),
                               str(round(mean_test, 2)) + ' +/- '
                               + str(round(stddev_test, 3))]

    # return different dataframes depending on what was asek
    if output_type == 'type':
        return rmse_df_26, rmse_df_35, rmse_df_44
    if output_type == 'site':
        return rmse_df_sub, rmse_df_int


########################################################
########################################################
# This is the wrapper df. Gives RMSE tables. Can do it by type and site
# by changing 'output_type'
def rfr_lasso_rmse(df, o=0, d_start=5, num_trees=100, max_feat='auto',
                   max_depth=5, min_samp_leaf=2, min_samples_split=5,
                   folds=5, label_type=1, label_site=4, output_type='none'):
    '''
    This is a wrapper func that performs RFR for energy levels in impurity
    doped sc. It returns a table with columns train RMSE and test RMSE. Each
    row in the table contain the train RMSE/test RMSE for a fold of cross
    validation. The parameters of the RFR model can be tuned with the
    arguments. By changing the 'output_type' arguement, the function can also
    return RMSE values for how well it predicts each type of sc in the data,
    or each defect site. The outputs it can predict are: formation enthalpy of
    the A-rich site/B-rich site (for AB semiconductor), and transition energy
    levels +3/+2 -> -2/-3. All together it predicts 8 outputs.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - o: int. column index of the output. Deafult: 0.
        - d_start: int. column index to that the descriptors columns start at.
        In the input df, the descriptors must start at some column at index
        df_start to the last column in the dataframe. Default: 5.
        - num_trees: int. Number of estimators (trees) to by used by the
        RFR in the random forest. Default:100.
        - max_feat: str. The number of features to consider when looking for
        the best split. Default: 'auto'
        - max_depth: int. The maximum depth of each tree in the forest.
        Keep this value low to mitigate overfitting. Default:5.
        - min_samp_leaf: int. The minimum number of samples required to be at
         a leaf node. Deafult: 2.
        - min_samples_split: int. The minimum number of samples required to
         split an internal node. Default: 5.
        - folds: int. Number of folds to to split the data in cross validation.
        Default: 5.
        - label_type: int. column index of "Type" column in DFT training
        dataframe. Default: 1.
        - label_site int. column index of "Site" column in DFT training
        dataframe. Dafault: 4.
        - output_type: str. Values must be 'none', 'type', 'site'. If 'none',
        then only returns one table of overall train/test RMSE. If 'type',
        returns 4 tables. The overall RMSE table, and a table for how the
        model predicts each type of sc. If 'site', returns 3 tables. overall
        and a table for RMSE of each defect site (sub, int).

    Outputs
        - rmse_df: panads df. Table where columns are train RMSE/ test RMSE
        and rows are RMSE values for each fold of cross validation for RFR
        prediction.
        - rmse_df_26/35/44: pandas df. same as rmse_df but specific to each
        type of sc.
        - rmse_df_ma/mb/mia/mib/mineut: pandas df. same as rmse_df but
        specific to each defect site.
    '''
    # make the dataframe stratifiable by type and site
    b = stratify_df(df, label_type, label_site)

    # identify the descriptor columns and output column
    X, y = descriptors_outputs(df, d_start, o)

    # establish the stratified k-fold cross validation, folds in an input
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=130)

    # establish the RFR classifier with certain parameters, paramters are inputs
    clf = RandomForestRegressor(n_estimators=num_trees, max_features=max_feat,
                                max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=2, random_state=130)

    # empty lists for RMSE values from each fold of CV to be added to
    # these lists are implemented for all output_types
    train_rmse_list = []
    test_rmse_list = []

    if output_type == 'none':
        # this will iterate as many times as determined by the input 'folds'
        # choosing training and testing points at random. Training set size is k-1
        # folds, and testing set size of 1 fold is "held-out"
        for train_idx, test_idx in skf.split(df, b):

            # updates the train and test RMSE lists with an RMSE from each fold of the CV
            train_rmse_list, test_rmse_list, _, _ = \
                rmse_total(df, X, y, train_idx, test_idx, clf,
                           train_rmse_list, test_rmse_list)

        # makes the train and test RMSE lists into a nice dataframe w/ mean and std dev
        rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)

        return rmse_df

    elif output_type == 'type':
        # establish default dictionaries for make_dict_tysi to add train/test
        # points by different sc type
        typetrain_dict = defaultdict(list)
        typetest_dict = defaultdict(list)

        for train_idx, test_idx in skf.split(df, b):
            # X and y_train get returned for fit_predict_tysi
            train_rmse_list, test_rmse_list, X_train, y_train = \
                        rmse_total(df, X, y, train_idx, test_idx, clf,
                                   train_rmse_list, test_rmse_list)

            # this func sorts the train and test points by sc type before training
            # and predicting with the model, and calculating RMSEs.
            typetrain_dict, typetest_dict = \
                wrapper_tysi(df, clf, X_train, y_train,
                             typetrain_dict, typetest_dict,
                             train_idx, test_idx, X, y, output_type)

        # make the RMSE information into nice dataframes
        rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)
        rmse_df_26, rmse_df_35, rmse_df_44 = \
            rmse_table_tysi(typetrain_dict, typetest_dict,
                            output_type)

        return rmse_df, rmse_df_26, rmse_df_35, rmse_df_44

    elif output_type == 'site':
        sitetrain_dict = defaultdict(list)
        sitetest_dict = defaultdict(list)

        for train_idx, test_idx in skf.split(df, b):
            train_rmse_list, test_rmse_list, X_train, y_train = \
                rmse_total(df, X, y, train_idx, test_idx, clf,
                           train_rmse_list, test_rmse_list)

            sitetrain_dict, sitetest_dict = \
                wrapper_tysi(df, clf, X_train, y_train, sitetrain_dict,
                             sitetest_dict, train_idx, test_idx, X, y,
                             output_type)

        rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)
        rmse_df_sub, rmse_df_int = rmse_table_tysi(sitetrain_dict,
                                                   sitetest_dict, output_type)

        return (rmse_df, rmse_df_sub, rmse_df_int)
    else:
        print("Invalid input for 'output_type'. Enter 'none', 'type', \
or 'site'.")
        return
###############################################################################
###############################################################################


def preds_iterator(X, y, train_idx, test_idx, clf, Y_train, Y_test,
                   PRED_train, PRED_test, Y_tr_idx, Y_te_idx):
    '''
    Creates 4 arrays: Y_train, Y_test, PRED_train, PRED_test, made up of y
    train data y test data, y train predictions, and y test predictions
    respectively. For each iteration of train/test split, a new group of
    points is added to each of these arrays.

    Inputs
        - X: pandas df. Dataframe with descriptors.
        - y: pandas df. Dataframe with output.
        - train_idx: np array. Indexes of training points.
        - test_idx: np array. Indexes of testing points.
        - clf: RandomForestRegressor from sklearn
        - Y_train: np array. empty array.
        - Y_test: np array. empty array.
        - PRED_train: np array. empty array.
        - PRED_test: np array. empty array.

    Outputs
        - Y_train: np array. array of y_train points from every iteration
        of train/test split via function traintest.
        - Y_test: np array. array of y_test points from every iteration
        of train/test split via function traintest.
        - PRED_train: np array. array of trainpred points from every iteration
        of train/test split via fit_predict.
        - PRED_test: np array. array of testpred points from every iteration
        of train/test split via fit_predict.
    '''
    # this function runs for n folds x n folds number of times. For each iteration:

    # get the train and test points
    X_train, X_test, y_train, y_test = traintest(X, y, train_idx, test_idx)

    # train model and predict on X_train and X_test data
    trainpred, testpred = fit_predict(X_train, y_train, X_test, clf)

    # store the y_train DFT points and indexes to np arrays
    # turn the arrays into a list so they can be used in plot_sorter funcs
    Y_train = list(np.append(Y_train, y_train))
    Y_tr_idx = list(np.append(Y_tr_idx, y_train.index))

    # store the y_test DFT points and indexes to np arrays
    Y_test = list(np.append(Y_test, y_test))
    Y_te_idx = list(np.append(Y_te_idx, y_test.index))

    # store the predicted train and test points to np arrays
    PRED_train = list(np.append(PRED_train, trainpred))
    PRED_test = list(np.append(PRED_test, testpred))



    return Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx


def zip_to_ddict(Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx):
    '''
    This function makes default dictionaries of keys that are DFT points, and
    values that are the predicted points across folds of CV (repeated folds
    num of times).

    Inputs
        - Y_train: list. DFT training points from the input df.
        - Y_test: list. DFT testing points from the input df.
        - PRED_train: list. predicted training points from the RFR clf.
        - PRED_test: list. predicted testing points from the RFR clf.

    Outputs
        - train: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times).
        - test_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)

    Notes: the "train" data comes from the training set in cross validation,
    and the "test" data comes from teh testing set. This is repeated for n
    number of folds * n folds times. Every point in the dataset is a train
    and test point.
    '''
    # this function runs for n folds x n folds number of times. For each iteration:

    # zip the training/testing DFT points with their indexes into tuples (idx, DFT)
    Y_trainidx_tuple = list(zip(Y_tr_idx, Y_train))
    Y_testidx_tuple = list(zip(Y_te_idx, Y_test))

    # zip the DFT/idx tuple with the predicted value for that point
    train_zip = list(zip(Y_trainidx_tuple, PRED_train))
    test_zip = list(zip(Y_testidx_tuple, PRED_test))

    # default dictionaries that store points with the same keys together
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    # add zipped trainng points to the train default dictionary
    for train_dft, train_preds in train_zip:
        train_dict[train_dft].append(train_preds)

    # add zipped testing points to the test default dictionary
    for test_dft, test_preds in test_zip:
        test_dict[test_dft].append(test_preds)

    return train_dict, test_dict


def dict_sorter(dic):
    '''
    Sorts default dictionaries which contain DFT vals as keys, and predicted
    vals from RFR cross validation as values into 3 lists. A list of keys,
    a list of the mean of values across the folds of cv, and a list of the std
    of the values across the folds of cv. Lists will be made into dataframes.

    Inputs:
        - dic: default dict. Keys are DFT values. Values are predicted values
        from cv during RFR.
    Outputs:
        - dft_list: list. keys from default dict.
        - mean_list: list. mean of the values associated with every key.
        - std_list list. std dev of the values associated with every key.
    '''
    # empty lists for each operation that will be performed on the values in the
    # train or test dictionaries
    idx_list = []
    dft_list = []
    mean_list = []
    std_list = []

    # go through the dictionary and put values into lists/perform operations to
    # be used laster
    for key, vals in dic.items():
        idx_list.append(key[0])
        dft_list.append(key[1])
        mean_list.append(mean(vals))
        std_list.append(stdev(vals))

    #return dft_list, mean_list, std_list
    return idx_list, dft_list, mean_list, std_list


def plotdict_tysi(test_dict, df, output_name, output_type):
    '''
    Sorts the test dictionary into separate dictionaries by 'type' or 'site',
    depending on output_type selected. The keys in the new dictionary are the
    DFT datapoints, the values are predicted points across the folds of cross
    validation.

    Inputs
        - test_dict: default dict. key are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - output_name: str. name of the column of output data.
        - output_type: str. Values must be 'type', 'site'.

    Outputs
        - test_dict(): dict. keys are dft points, values are predicted points
        across folds of CV (repeated folds num of times). The test dict
        is seperated by sc type or defect site type. See notes.

    Notes: If output = 'type', returns 3 dicts (one for each sc type).
    If output = 'site', returns 2 dict (one for each type of defect
    site (substitutional/interstitial)).
    '''
    if output_type == 'type':
        # empty dictionaries to store data from different sc types in
        test_dict26 = {}
        test_dict35 = {}
        test_dict44 = {}

        for key, vals in test_dict.items():
            # a DFT's sc type is identified by the 'Type' at the index of the DFT point
            # which was stored with the point in preds_iterator
            idx = key[0]
            if (df['Type'].loc[idx] == 'II-VI'):
                test_dict26.update({key: vals})
            elif (df['Type'].loc[idx] == 'III-V'):
                test_dict35.update({key: vals})
            elif (df['Type'].loc[idx] == 'IV-IV'):
                test_dict44.update({key: vals})

        return test_dict26, test_dict35, test_dict44

    if output_type == 'site':
        test_dictsub = {}
        test_dictint = {}

        for key, vals in test_dict.items():
            idx = key[0]
            if (df['Site'].loc[idx] == 'M_A' or
                    df['Site'].loc[idx] == 'M_B'):
                test_dictsub.update({key: vals})
            elif (df['Site'].loc[idx] == 'M_i_A' or
                  df['Site'].loc[idx] == 'M_i_B' or
                  df['Site'].loc[idx] == 'M_i_neut'):
                test_dictint.update({key: vals})

        return test_dictsub, test_dictint


def plot_sorter_none(train_dict, test_dict, output_name, show_plot):
    '''
    Plots the train and test dictionary points as a parity plot. Returns
    training and testing tables that include the dft point, mean predicted
    point fromt the model, and std dev.

    Inputs
        - train_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times).
        - test_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)
        - output_name: str. name of the column of output data.

    Outputs
        - train_dataframe: pd dataframe. Data from the "training set"
        during CV. Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data.
        - test_dataframe: pd dataframe. Data from the "testing set"
        during CV Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data.
    '''
    # get the DFT points, mean and std dev of prediction points into separate lists
    #dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    #dft_test, mean_test, stddev_test = dict_sorter(test_dict)

    idx_train, dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    idx_test, dft_test, mean_test, stddev_test = dict_sorter(test_dict)

    # make these values into nice dataframes.
    #train_dataframe = \
    #    pd.DataFrame(data={dft_train': dft_train, 'mean_train': mean_train,
    #                 'stddev_train': stddev_train})
    #test_dataframe = \
    #    pd.DataFrame(data={'dft_test': dft_test, 'mean_test': mean_test,
    #                 'stddev_test': stddev_test})
    train_dataframe = \
        pd.DataFrame(data={'idx_train': idx_train, 'dft_train': dft_train, 'mean_train': mean_train,
                     'stddev_train': stddev_train})
    test_dataframe = \
        pd.DataFrame(data={'idx_test': idx_test, 'dft_test': dft_test, 'mean_test': mean_test,
                     'stddev_test': stddev_test})

    # parity plot with uncertainties
    if show_plot is True:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(dft_train, mean_train, yerr=stddev_train, alpha=0.5,
                    label='train', color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test, mean_test, yerr=stddev_test, alpha=0.5,
                    label='test', color='red', zorder=3, fmt='o', markersize=4)
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
    else:
        pass

    return train_dataframe, test_dataframe


def plot_sorter_type(train_dict, test_dict, output_name, df, output_type,
                     show_plot):
    '''
    Plots the train and test dictionary points as a parity plot. The test
    points are plotted by type.

    Inputs
        - train_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times).
        - test_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)
        - output_name: str. name of the column of output data.
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - output_type: str. Values must be 'type', 'site'.

    Outputs
        - train_dataframe: pd dataframe. Data from the "training set"
        during CV. Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data.
        - test_dataframe_26/35/44: pd dataframe. Data from the "testing set"
        during CV Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data. Each
        table only contains the points from their respective sc type.
        4 tables total in output.
    '''
    test_dict26, test_dict35, test_dict44 = \
        plotdict_tysi(test_dict, df, output_name, output_type)

    # get the DFT points, mean and std dev of prediction points into separate lists
    dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    dft_test26, mean_test26, stddev_test26 = dict_sorter(test_dict26)
    dft_test35, mean_test35, stddev_test35 = dict_sorter(test_dict35)
    dft_test44, mean_test44, stddev_test44 = dict_sorter(test_dict44)

    train_dataframe = \
        pd.DataFrame(data={'dft_train': dft_train, 'mean_train': mean_train,
                     'stddev_train': stddev_train})
    test26_dataframe = \
        pd.DataFrame(data={'dft_test_26': dft_test26,
                     'mean_test_26': mean_test26,
                           'stddev_test_26': stddev_test26})
    test35_dataframe = \
        pd.DataFrame(data={'dft_test_35': dft_test35,
                     'mean_test_35': mean_test35,
                           'stddev_test_35': stddev_test35})
    test44_dataframe = \
        pd.DataFrame(data={'dft_test_44': dft_test44,
                     'mean_test_44': mean_test44,
                           'stddev_test_44': stddev_test44})

    # parity plot with uncertainties
    if show_plot is True:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(dft_train, mean_train, yerr=stddev_train, alpha=0.5,
                    label='train', color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test26, mean_test26, yerr=stddev_test26, alpha=0.5,
                    label='II-VI test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test35, mean_test35, yerr=stddev_test35, alpha=0.5,
                    label='III-V test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test44, mean_test44, yerr=stddev_test44, alpha=0.5,
                    label='IV-IV test', zorder=3, fmt='o', markersize=4)
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
    else:
        pass

    return (train_dataframe, test26_dataframe, test35_dataframe,
            test44_dataframe)


def plot_sorter_site(train_dict, test_dict, output_name, df, output_type,
                     show_plot):
    '''
    Plots the train and test dictionary points as a parity plot. The test
    points are plotted by defect site (sub/int).

    Inputs
        - train_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times).
        - test_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)
        - output_name: str. name of the column of output data.
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - output_type: str. Values must be 'type', 'site'.

    Outputs
        - train_dataframe: pd dataframe. Data from the "training set"
        during CV. Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data.
        - test_dataframe_sub/int: pd dataframe. Data from the "testing set"
        during CV Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data. Each
        table only contains the points from their respective defect site.
        3 tables total in output.
    '''
    test_dictsub, test_dictint = plotdict_tysi(test_dict, df,
                                               output_name, output_type)

    dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    dft_test_sub, mean_test_sub, stddev_test_sub = dict_sorter(test_dictsub)
    dft_test_int, mean_test_int, stddev_test_int = dict_sorter(test_dictint)

    train_dataframe = \
        pd.DataFrame(data={'dft_train': dft_train, 'mean_train': mean_train,
                     'stddev_train': stddev_train})
    testsub_dataframe = \
        pd.DataFrame(data={'dft_test_sub': dft_test_sub,
                     'mean_test_sub': mean_test_sub,
                           'stddev_test_sub': stddev_test_sub})
    testint_dataframe = \
        pd.DataFrame(data={'dft_test_int': dft_test_int,
                     'mean_test_int': mean_test_int,
                           'stddev_test_int': stddev_test_int})

    # parity plot with uncertainties
    if show_plot is True:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(dft_train, mean_train, yerr=stddev_train, alpha=0.5,
                    label='train', color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test_sub, mean_test_sub, yerr=stddev_test_sub, alpha=0.5,
                    label='sub site test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(dft_test_int, mean_test_int, yerr=stddev_test_int, alpha=0.5,
                    label='int site test', zorder=3, fmt='o', markersize=4)

        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
    else:
        pass

    return (train_dataframe, testsub_dataframe, testint_dataframe)


def rfr_lasso_plot(df, o=0, d_start=5, num_trees=100, max_feat='auto',
                   max_depth=5, min_samp_leaf=2, min_samples_split=5, folds=5,
                   label_type=1, label_site=4, output_type='none',
                   show_plot=True):
    '''
    Wrapper func that performs RFR and plots the predicted train
    and test points vs the true train and test points (parity plot),with
    uncertainty. Uncertainty comes from n-folds of cross validation
    performed n-folds time. Tables are returned that have the actual
    dft points, the mean of the predicted points, and the std across the
    folds of cross validation. RMSE tables are returned that give a metric for
    how the model is predicting on a particular data set. See Notes section
    to understand how the 'output_type' selection changes number of outputs.

    Inputs
        - df: pandas df. A ML training dataset that contains targets and
        features.
        - o: int. column index of the output. Deafult: 0.
        - d_start: int. column index to that the descriptors columns start at.
        In the input df, the descriptors must start at some column at index
        df_start to the last column in the dataframe. Default: 3.
        - num_trees: int. Number of estimators (trees) to by used by the
        RFR in the random forest. Default:100.
        - max_feat: str. The number of features to consider when looking for
        the best split. Default: 'auto'
        - max_depth: int. The maximum depth of each tree in the forest.
        Keep this value low to mitigate overfitting. Default:5.
        - min_samp_leaf: int. The minimum number of samples required to be at
         a leaf node. Deafult: 2.
        - min_samples_split: int. The minimum number of samples required to
         split an internal node. Default: 5.
        - folds: int. Number of folds to to split the data in cross validation.
        Default: 5.
        - label_type: int. column index of "Type" column in DFT training
        dataframe. Default: 1.
        - label_site int. column index of "Site" column in DFT training
        dataframe. Dafault: 4.
        - output_type: str. Values must be 'none', 'type', 'site'.
        - show_plot: bool. If True, plot displays, if False, plot does not.

    Outputs
        - rmse_df: panads df. Table where columns are train RMSE/ test RMSE
        and rows are RMSE values for each fold of cross validation for RFR
        prediction.
        - rmse_df_26/35/44: pandas df. same as rmse_df but specific to each
        type of sc.
        - rmse_df_sub/int: pandas df. same as rmse_df but specific to each
        type of defect site.
        - train_dataframe: pd dataframe. Data from the "training set"
        during CV. Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data.
        - test_dataframe_26/35/44: pd dataframe. Data from the "testing set"
        during CV Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data. Each
        table only contains the points from their respective sc type.
        4 tables total in output.
        - test_dataframe_sub/int: pd dataframe. Data from the "testing set"
        during CV Columns: dft data, mean of predicted
        data(across folds of cv), std deviated of predicted data. Each
        table only contains the points from their respective defect site.
        3 tables total in output.

    Notes: If 'none', func returns 3 tables: train/test predictions tables and
    rmse table. If 'type', func returns 8 tables: (train, test x3 for each sc
    type, and rmse table + rmse table for each sc type). If 'site', returns 6
    tables (train, test x2 for each defect type, and rmse table + rmse table
    for each defect site type).
    '''
    b = stratify_df(df, label_type, label_site)

    X, y = descriptors_outputs(df, d_start, o)

    output_name = df.columns[o]

    #kf = KFold(n_splits=folds, shuffle=True)
    skf_plot = StratifiedKFold(n_splits=folds, shuffle=True)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=130)

    clf = RandomForestRegressor(n_estimators=num_trees, max_features=max_feat,
                                max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1, random_state=130)

    Y_train = np.empty(0)
    Y_test = np.empty(0)
    PRED_train = np.empty(0)
    PRED_test = np.empty(0)

    Y_tr_idx = np.empty(0)
    Y_te_idx = np.empty(0)


    for i in tqdm(range(folds)):

        #for train_idx, test_idx in kf.split(X, y):
        for train_idx, test_idx in skf_plot.split(df, b):

            (Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx) = \
                preds_iterator(X, y, train_idx, test_idx, clf, Y_train, Y_test,
                               PRED_train, PRED_test, Y_tr_idx, Y_te_idx)

    train_dict, test_dict = \
        zip_to_ddict(Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx)

    train_rmse_list = []
    test_rmse_list = []

    for train_idx, test_idx in skf.split(df, b):

        train_rmse_list, test_rmse_list, _, _ = \
            rmse_total(df, X, y, train_idx, test_idx, clf,
                       train_rmse_list, test_rmse_list)

    rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)

    if output_type == 'none':
        train_df, test_df = \
            plot_sorter_none(train_dict, test_dict, output_name, show_plot)

        return train_df, test_df, rmse_df

    elif output_type == 'type':
        (train_df, test26_df, test35_df, test44_df) = \
            plot_sorter_type(train_dict, test_dict, output_name, df,
                             output_type, show_plot)

        typetrain_rmse_dict = defaultdict(list)
        typetest_rmse_dict = defaultdict(list)

        for train_idx, test_idx in skf.split(df, b):
            X_train, _,  y_train, _ = \
                traintest(X, y, train_idx, test_idx)

            typetrain_rmse_dict, typetest_rmse_dict = \
                wrapper_tysi(df, clf, X_train, y_train,
                             typetrain_rmse_dict, typetest_rmse_dict,
                             train_idx, test_idx, X, y, output_type)

        rmse_df_26, rmse_df_35, rmse_df_44 = \
            rmse_table_tysi(typetrain_rmse_dict, typetest_rmse_dict,
                            output_type)

        return (train_df, test26_df, test35_df, test44_df, rmse_df, rmse_df_26,
                rmse_df_35, rmse_df_44)

    elif output_type == 'site':
        (train_df, testsub_df, testint_df) = \
            plot_sorter_site(train_dict, test_dict, output_name, df,
                             output_type, show_plot)

        sitetrain_rmse_dict = defaultdict(list)
        sitetest_rmse_dict = defaultdict(list)

        for train_idx, test_idx in skf.split(df, b):
            X_train, _,  y_train, _ = \
                traintest(X, y, train_idx, test_idx)

            sitetrain_rmse_dict, sitetest_rmse_dict = \
                wrapper_tysi(df, clf, X_train, y_train, sitetrain_rmse_dict,
                             sitetest_rmse_dict, train_idx, test_idx, X, y,
                             output_type)

        rmse_df_sub, rmse_df_int = \
            rmse_table_tysi(sitetrain_rmse_dict, sitetest_rmse_dict,
                            output_type)

        return (train_df, testsub_df, testint_df, rmse_df, rmse_df_sub,
                rmse_df_int)
