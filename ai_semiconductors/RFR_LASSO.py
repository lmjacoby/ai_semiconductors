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
    clf.fit(X_train, y_train)

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
    train_rmse = mean_squared_error(y_train, trainpred, squared=False)
    test_rmse = mean_squared_error(y_test, testpred, squared=False)

    return train_rmse, test_rmse


def rmse_list(train_rmse_list, test_rmse_list, train_rmse, test_rmse):
    '''
    This function appends the train/test rmse from each fold in the CV to
    a list so that each CV round can be analyzed.

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

    X_train, X_test, y_train, y_test = traintest(X, y, train_idx, test_idx)

    trainpred, testpred = fit_predict(X_train, y_train, X_test, clf)

    train_rmse, test_rmse = rmse(y_train, y_test, trainpred, testpred)

    train_rmse_list, test_rmse_list = rmse_list(train_rmse_list,
                                                test_rmse_list,
                                                train_rmse, test_rmse)

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

    d = {'train rmse': train_rmse_list,
         'test rmse': test_rmse_list}

    rmse_df = pd.DataFrame(data=d)

    mean_train = rmse_df[rmse_df.columns[0]].mean()
    stddev_train = rmse_df[rmse_df.columns[0]].std()
    mean_test = rmse_df[rmse_df.columns[1]].mean()
    stddev_test = rmse_df[rmse_df.columns[1]].std()

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
    if output_type == 'type':

        train_idx_26 = []
        train_idx_35 = []
        train_idx_44 = []
        for idx in list(train_idx):
            if df['Type'].iloc[idx] == 'II-VI':
                train_idx_26.append(idx)
            elif df['Type'].iloc[idx] == 'III-V':
                train_idx_35.append(idx)
            elif df['Type'].iloc[idx] == 'IV-IV':
                train_idx_44.append(idx)

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

        return (train_idx_26, train_idx_35, train_idx_44, test_idx_26,
                test_idx_35, test_idx_44)

    if output_type == 'site':

        train_idx_sub = []
        train_idx_int = []
        for idx in list(train_idx):
            if df['Site'].iloc[idx] == 'M_A' or df['Site'].iloc[idx] == 'M_B':
                train_idx_sub.append(idx)
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

        return (train_idx_sub, train_idx_int, test_idx_sub, test_idx_int)


def traintest_df_tysi(X, y, output_type, *args):
    '''
    This function takes the indexes of each type of sc or site of defect
    obtained in 'df_tysi'and creates an ordered dict where vals are 4
    different dataframes (X_train, X_test, y_train, y_test), for each of the
    three types of semiconductors (12 dataframes total) or each of the two
    types of defect sites (8 dataframes total).

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

        num_list = [26, 35, 44]
        groups = [(args[0], args[3]), (args[1], args[4]),
                  (args[2], args[5])]

        p = OrderedDict()

        for (num, grp) in zip(num_list, groups):
            p['X_train_{0}'.format(num)] = X.iloc[grp[0]]
            p['X_test_{0}'.format(num)] = X.iloc[grp[1]]
            p['y_train_{0}'.format(num)] = y.iloc[grp[0]]
            p['y_test_{0}'.format(num)] = y.iloc[grp[1]]

    if output_type == 'site':
        site_list = ['sub', 'int']
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
    if output_type == 'type':
        a = (tuple(p.values())[:4])
        b = (tuple(p.values())[4:8])
        c = (tuple(p.values())[8:12])

        xy_list = [a, b, c]

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
        - val: tuple. tuple from xytype_list or xysite_list
        - X_train: np array. descriptor values of training data set.
        - y_train: np array. output values of training data set.
    Outputs
        - trainpred: np array. predicted output value for every point in
        the train data set (for each type or site).
        - testpred: np array. predicted output value for every point in
        the test data set (for each type or site).
    '''
    clf.fit(X_train, y_train)

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
        - val: tuple. tuple from xytype_list or xysite_list
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

    index = val[0].index.tolist()
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
    train_dict[key_train].append(train_rmse)
    test_dict[key_test].append(test_rmse)

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
    if output_type == 'type':
        (train_idx_26, train_idx_35, train_idx_44, test_idx_26,
         test_idx_35, test_idx_44) = df_tysi(df, train_idx, test_idx,
                                             output_type)

        p = traintest_df_tysi(X, y, output_type, train_idx_26, train_idx_35,
                              train_idx_44, test_idx_26, test_idx_35,
                              test_idx_44)

    elif output_type == 'site':
        (train_idx_sub, train_idx_int, test_idx_sub,
         test_idx_int) = df_tysi(df, train_idx, test_idx,
                                 output_type)

        p = traintest_df_tysi(X, y, output_type, train_idx_sub, train_idx_int,
                              test_idx_sub, test_idx_int)

    xy_list = make_xy_list(p, output_type)

    for xy in xy_list:

        trainpred, testpred = fit_predict_tysi(clf, xy,
                                               X_train, y_train)

        train_rmse, test_rmse = rmse_tysi(xy, trainpred, testpred)

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
    train_rmse_df = pd.DataFrame.from_dict(train_dict)
    test_rmse_df = pd.DataFrame.from_dict(test_dict)

    if output_type == 'type':

        rmse_df_26 = pd.concat([train_rmse_df['train rmse II-VI'],
                               test_rmse_df['test rmse II-VI']], axis=1)
        rmse_df_35 = pd.concat([train_rmse_df['train rmse III-V'],
                               test_rmse_df['test rmse III-V']], axis=1)
        rmse_df_44 = pd.concat([train_rmse_df['train rmse IV-IV'],
                               test_rmse_df['test rmse IV-IV']], axis=1)

        df_list = [rmse_df_26, rmse_df_35, rmse_df_44]

    if output_type == 'site':

        rmse_df_sub = pd.concat([train_rmse_df['train rmse sub'],
                                test_rmse_df['test rmse sub']], axis=1)
        rmse_df_int = pd.concat([train_rmse_df['train rmse int'],
                                test_rmse_df['test rmse int']], axis=1)

        df_list = [rmse_df_sub, rmse_df_int]

    for dtfr in df_list:
        mean_train = dtfr[dtfr.columns[0]].mean()
        stddev_train = dtfr[dtfr.columns[0]].std()
        mean_test = dtfr[dtfr.columns[1]].mean()
        stddev_test = dtfr[dtfr.columns[1]].std()

        dtfr.loc[len(dtfr)] = [str(round(mean_train, 2)) + ' +/- '
                               + str(round(stddev_train, 3)),
                               str(round(mean_test, 2)) + ' +/- '
                               + str(round(stddev_test, 3))]

    if output_type == 'type':
        return rmse_df_26, rmse_df_35, rmse_df_44
    if output_type == 'site':
        return rmse_df_sub, rmse_df_int


########################################################
########################################################
# This is the wrapper df. Gives RMSE tables. Can do it by type and site
# by changing 'output_type'
def RFR_LASSO_rmse(df, o=0, d_start=5, num_trees=100, max_feat='auto',
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
        - label_type: int. column index of "Type" column. Default: 1.
        - label_site int. column index of "Site" column. Dafault: 4.
        - output_type: str. Values must be 'none', 'type', 'site'. If 'none',
        then only returns one table of overall train/test RMSE. If 'type',
        returns 4 tables. The overall RMSE table, and a table for how the
        model predicts each type of sc. If 'site', returns 6 tables. overall
        and a table for how the model predicts each defect site.

    Outputs
        - rmse_df: panads df. Table where columns are train RMSE/ test RMSE
        and rows are RMSE values for each fold of cross validation for RFR
        prediction.
        - rmse_df_26/35/44: pandas df. same as rmse_df but specific to each
        type of sc.
        - rmse_df_ma/mb/mia/mib/mineut: pandas df. same as rmse_df but
        specific to each defect site.
    '''

    b = stratify_df(df, label_type, label_site)

    X, y = descriptors_outputs(df, d_start, o)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=130)

    clf = RandomForestRegressor(n_estimators=num_trees, max_features=max_feat,
                                max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=2, random_state=130)

    train_rmse_list = []
    test_rmse_list = []

    if output_type == 'none':
        for train_idx, test_idx in skf.split(df, b):

            train_rmse_list, test_rmse_list, _, _ = rmse_total(df, X, y,
                                                               train_idx,
                                                               test_idx, clf,
                                                               train_rmse_list,
                                                               test_rmse_list)

        rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)

        return rmse_df

    elif output_type == 'type':
        typetrain_dict = defaultdict(list)
        typetest_dict = defaultdict(list)

        for train_idx, test_idx in skf.split(df, b):
            train_rmse_list, test_rmse_list, X_train, y_train = \
                        rmse_total(df, X, y, train_idx, test_idx, clf,
                                   train_rmse_list, test_rmse_list)

            typetrain_dict, typetest_dict = \
                wrapper_tysi(df, clf, X_train, y_train,
                             typetrain_dict, typetest_dict,
                             train_idx, test_idx, X, y, output_type)

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
    dft_list = []
    mean_list = []
    std_list = []

    for key, vals in dic.items():
        dft_list.append(key)
        mean_list.append(mean(vals))
        std_list.append(stdev(vals))

    return dft_list, mean_list, std_list


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
        test_dict26 = {}
        test_dict35 = {}
        test_dict44 = {}

        for key, vals in test_dict.items():
            idx = df[df[output_name] == key].index
            if (df['Type'].loc[idx].any() == 'II-VI'):
                test_dict26.update({key: vals})
            elif (df['Type'].loc[idx].any() == 'III-V'):
                test_dict35.update({key: vals})
            elif (df['Type'].loc[idx].any() == 'IV-IV'):
                test_dict44.update({key: vals})

        return test_dict26, test_dict35, test_dict44

    if output_type == 'site':
        test_dictsub = {}
        test_dictint = {}

        for key, vals in test_dict.items():
            idx = df[df[output_name] == key].index
            if (df['Site'].loc[idx].any() == 'M_A' or
                    df['Site'].loc[idx].any() == 'M_B'):
                test_dictsub.update({key: vals})
            elif (df['Site'].loc[idx].any() == 'M_i_A' or
                  df['Site'].loc[idx].any() == 'M_i_B' or
                  df['Site'].loc[idx].any() == 'M_i_neut'):
                test_dictint.update({key: vals})

        return test_dictsub, test_dictint


def zip_to_ddict(Y_train, Y_test, PRED_train, PRED_test):
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

    train_zip = list(zip(Y_train, PRED_train))
    test_zip = list(zip(Y_test, PRED_test))

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for train_dft, train_preds in train_zip:
        train_dict[train_dft].append(train_preds)

    for test_dft, test_preds in test_zip:
        test_dict[test_dft].append(test_preds)

    return train_dict, test_dict


def plot_sorter_none(train_dict, test_dict, output_name):
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

    dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    dft_test, mean_test, stddev_test = dict_sorter(test_dict)

    train_dataframe = \
        pd.DataFrame(data={'dft_train': dft_train, 'mean_train': mean_train,
                     'stddev_train': stddev_train})
    test_dataframe = \
        pd.DataFrame(data={'dft_test': dft_test, 'mean_test': mean_test,
                     'stddev_test': stddev_test})

    # parity plot with uncertainties
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

    return train_dataframe, test_dataframe


def plot_sorter_type(train_dict, test_dict, output_name, df, output_type):
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

    return (train_dataframe, test26_dataframe, test35_dataframe,
            test44_dataframe)


def plot_sorter_site(train_dict, test_dict, output_name, df, output_type):
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

    return (train_dataframe, testsub_dataframe, testint_dataframe)


# updated 6/9/2020
def RFR_LASSO_plot(df, o=0, d_start=3, num_trees=100, max_feat='auto',
                   max_depth=5, min_samp_leaf=2, min_samples_split=5, folds=5,
                   output_type='none'):
    '''
    This is a wrapper func that performs RFR and plots the predicted train
    and test points vs the true train and test points (parity plot),with
    uncertainty. Uncertainty comes from n-folds of cross validation
    performed n-folds time. Tables are also returned that have the actual
    dft points, the mean of the predicted points, and the std across the
    folds of cross validation.

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
        - output_type: str. Values must be 'none', 'type', 'site'.

    Outputs
        - rmse_df: panads df. Table where columns are train RMSE/ test RMSE
        and rows are RMSE values for each fold of cross validation for RFR
        prediction.
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

    Notes:  If 'none', then returns train/test table and rmse table. If
    'type', returns 5 tables (train, test x3 for each sc type, and rmse table).
     If 'site', returns 4 tables (train, test x2 for each defect type, and
     rmse table).
    '''
    X, y = descriptors_outputs(df, d_start, o)

    output_name = df.columns[o]

    kf = KFold(n_splits=folds, shuffle=True)

    clf = RandomForestRegressor(n_estimators=num_trees, max_features=max_feat,
                                max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1, random_state=130)

    Y_train = np.empty(0)
    Y_test = np.empty(0)
    PRED_train = np.empty(0)
    PRED_test = np.empty(0)

    train_rmse_list = []
    test_rmse_list = []

    for i in tqdm(range(folds)):

        for train_idx, test_idx in kf.split(X, y):

            X_train, X_test, y_train, y_test = \
                traintest(X, y, train_idx, test_idx)

            trainpred, testpred = fit_predict(X_train, y_train, X_test, clf)

            train_rmse, test_rmse = rmse(y_train, y_test, trainpred, testpred)

            train_rmse_list, test_rmse_list = \
                rmse_list(train_rmse_list, test_rmse_list, train_rmse,
                          test_rmse)

            Y_train = list(np.append(Y_train, y_train))
            Y_test = list(np.append(Y_test, y_test))
            PRED_train = list(np.append(PRED_train, trainpred))
            PRED_test = list(np.append(PRED_test, testpred))

    train_dict, test_dict = \
        zip_to_ddict(Y_train, Y_test, PRED_train, PRED_test)

    rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)

    if output_type == 'none':
        (train_dataframe, test_dataframe) = \
            plot_sorter_none(train_dict, test_dict, output_name)

        return train_dataframe, test_dataframe, rmse_df

    elif output_type == 'type':
        (train_dataframe, test26_dataframe, test35_dataframe,
         test44_dataframe) = plot_sorter_type(train_dict, test_dict,
                                              output_name, df, output_type)
        return (train_dataframe, test26_dataframe, test35_dataframe,
                test44_dataframe, rmse_df)

    elif output_type == 'site':
        (train_dataframe, testsub_dataframe,
         testint_dataframe) = plot_sorter_site(train_dict, test_dict,
                                               output_name, df, output_type)
        return (train_dataframe, testsub_dataframe,
                testint_dataframe, rmse_df)
