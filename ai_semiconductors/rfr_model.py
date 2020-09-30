import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statistics import mean, stdev
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

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

    # append a line ot the end of each dataframe with the mean and std
    # dev of the vals

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


def join_pred_labels(df_pred, df):
    '''
    This function appends the type/site/compound/impurity columns onto the
    prediction dataframes. Used in rfr_predictor and rfr_pp_predictor.

    Inputs
        - df_pred: pd df. prediction dataframe from rfr_predictor.
        - df: pd df. original dataframe with descriptors and DFT values

    Outputs
        - final_df: pandas df. Dataframe with complete (type/site/compound
        /impurity) information for each point in the df_pred dataframe.
    '''
    # use the index column from prediction df to find the same index points
    # in the original DFT dataframe. This df should be in the same order as the
    # prediction df, and contains type/site/compound/dopants/descriptors
    idx_df = df.loc[df_pred[df_pred.columns[0]]]

    # only use columns idx 1-4 of this new dataframe created to get the
    # type/ab/site/defect of every point
    abbr_idx_df = idx_df[idx_df.columns[1:5]]

    # index has to be reset so it can be cocatenated
    abbr_idx_df.reset_index(inplace=True)

    # make a copy of the new df so it doesn't cause a slicing error
    copy_abbr_idx_df = abbr_idx_df.copy()

    # a column named index was created from reset_index, and this contains
    # all the actual indexes. drop this, it's not needed.
    copy_abbr_idx_df.drop('index', axis=1, inplace=True)

    # concatenate the prediction dataframe with the df containnig
    # type/ab/site/defect
    final_df = pd.concat([copy_abbr_idx_df, df_pred], axis=1)

    # drop the index column that came from the prediction dataframe
    final_df.drop(final_df.columns[4], axis=1, inplace=True)

    return final_df


def add_site_col(dict_df):
    '''
    This function adds the column 'Site2' to a dataframe with a column 'Site'.
    'Site2' values are based off the values in 'Site', where if the 'Site'
    value == 'M_A' or 'M_B', 'Site2' value is 'sub'. If 'Site' value ==
    'M_i_A', 'M_i_B', or 'M_i_neut', 'Site2' value is 'int'.

    Inputs
        - dict_df: pandas df. Comes from the dictionary of dataframes in
        rmse_predictor.

    Outputs
        - dict_df: pandas df. Same as input dict, now with new column 'Site2'
    '''
    dict_df['Site2'] = \
        dict_df['Site'].map(lambda x: 'sub' if 'M_A' in x
                            else 'sub' if 'M_B' in x
                            else 'int' if 'M_i_A' in x
                            else 'int' if 'M_i_B' in x
                            else 'int' if 'M_i_neut' in x else '')
    return dict_df


########################################################
########################################################
def rfr_predictor(df, o=0, d_start=5, num_trees=100, max_feat='auto',
                  max_depth=5, min_samp_leaf=2, min_samples_split=5,
                  folds=5, label_type=1, label_site=4):
    '''
    This is a wrapper func that performs RFR with cross validation on a set of
    data with observed values and descriptors. For each  fold of CV, points
    are predicted for the train and test data. The function returns train and
    test dictionaries that contain the fold of CV as the key, and the a
    dataframe of type, ab, impurity, site, dft values, predicted values for
    every point in the train or test set in that fold.

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

    Outputs
        - folds_dict_train: pandas df. key is CV fold, value is dataframe of
         type, ab, impurity, site, dft values, predicted values for every
         point in the train set in that fold.
        - folds_dict_test: pandas df.  key is CV fold, value is dataframe of
         type, ab, impurity, site, dft values, predicted values for every
         point in the test set in that fold.
    '''
    # make the dataframe stratifiable by type and site
    b = stratify_df(df, label_type, label_site)

    # identify the descriptor columns and output column
    X, y = descriptors_outputs(df, d_start, o)

    # establish the stratified k-fold cross validation, folds in an input
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=130)

    # establish the RFR classifier with certain parameters, which are inputs
    clf = RandomForestRegressor(n_estimators=num_trees, max_features=max_feat,
                                max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=2, random_state=130)

    folds_dict_train = {}
    folds_dict_test = {}

    i = 1
    for train_idx, test_idx in skf.split(df, b):

        X_train, X_test, y_train, y_test = traintest(X, y, train_idx, test_idx)

        trainpred, testpred = fit_predict(X_train, y_train, X_test, clf)

        rmse_train = pd.DataFrame(data={'idx': y_train.index,
                                        'dft_train': y_train,
                                        'pred_train': trainpred})
        rmse_test = pd.DataFrame(data={'idx': y_test.index,
                                       'dft_test': y_test,
                                       'pred_test': testpred})

        rmse_list = [rmse_train, rmse_test]

        for rmse in rmse_list:
            rmse.reset_index(inplace=True)
            rmse.drop('index', axis=1, inplace=True)

        rmse_train_df = join_pred_labels(rmse_train, df)
        rmse_test_df = join_pred_labels(rmse_test, df)

        # key is the fold number, value is the dataframe of all the predicted
        # points (rmse_train or rmse_test)
        folds_dict_train[i] = rmse_train_df
        folds_dict_test[i] = rmse_test_df

        i += 1

    return folds_dict_train, folds_dict_test


def rmse_calculator(folds_dict_train, folds_dict_test, output_type):
    '''
    This function takes a dictionary of dataframes (from rfr_predictor), and
    computes the RMSE of the model at each fold in the CV, returning a df.
    output_type determines if the RMSE calculation is calculated for the
    overall dataset or if the RMSE is broken down by sc type ('type')
    or defect site ('site'). 'none' returns one df, 'type' returns three dfs,
    'site' returns 'two' dfs.

    Inputs
        - folds_dict_train: pandas df. key is CV fold, value is dataframe of
         type, ab, impurity, site, dft values, predicted values for every
         point in the train set in that fold.
        - folds_dict_test: pandas df.  key is CV fold, value is dataframe of
         type, ab, impurity, site, dft values, predicted values for every
         point in the test set in that fold.
        - output_type: str. 'none', 'type', or 'site'

    Outputs
        - rmse_df: pandas df. Table of train/test rmse values where each row
        is rmse for a fold in the CV. An average of all folds is the last row.

    Notes: If output_type is 'type', 3 rmse tables are returned broken down
    by: II-VI sc type, III-V sc type, IV-IV sc type. if output_type is 'site',
    2 rmse tables are returned broken down by: substitutional site,
    interstitial site.
    '''
    train_rmse_list = []
    test_rmse_list = []

    if output_type == 'none':

        # iterate through every entry in the dictionary and add rmse values
        # to train_rmse_list and test_rmse_list
        for i in range(len(folds_dict_train)):
            # dft train points
            y_train = folds_dict_train[i+1]['dft_train']

            # predicted train points
            trainpred = folds_dict_train[i+1]['pred_train']

            # dft test points
            y_test = folds_dict_test[i+1]['dft_test']

            # predicted test points
            testpred = folds_dict_test[i+1]['pred_test']

            # calculate the train and test rmse with rmse func
            train_rmse, test_rmse = rmse(y_train, y_test, trainpred, testpred)

            # add train and test rmse to their respective lists
            train_rmse_list, test_rmse_list = \
                rmse_list(train_rmse_list, test_rmse_list, train_rmse,
                          test_rmse)

        # make a table of the values
        rmse_df = rmse_table_ms(train_rmse_list, test_rmse_list)

        return rmse_df

    if output_type == 'type':

        # set up empty default dictionaries
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)

        # iteratively add values into the default dictionaries
        for i in range(len(folds_dict_train)):

            # make each dict from rmse_predictor into a tuple of 3 tuples
            # (sc type (str), df of values associated with that sc type)
            # (('II-VI', df), ('III-V', df), (IV-IV, df))
            # do this step so the dfs can be called.
            type_split_train = (tuple(folds_dict_train[i+1].groupby('Type')))

            type_split_test = (tuple(folds_dict_test[i+1].groupby('Type')))

            # go through each tuple (in the tuple of tuples) and extract
            # y_train, trainpred, etc...
            for type_idx in range(len(type_split_train)):

                # 1 to call the dataframe and not the sc type
                y_train = type_split_train[type_idx][1]['dft_train']
                trainpred = type_split_train[type_idx][1]['pred_train']

                y_test = type_split_test[type_idx][1]['dft_test']
                testpred = type_split_test[type_idx][1]['pred_test']

                # calculate the train and test rmse with rmse func
                train_rmse, test_rmse = \
                    rmse(y_train, y_test, trainpred, testpred)

                # key for ddict comes from first value in tuple
                type_key = type_split_train[type_idx][0]

                # append the train and test rmse to the ddict
                train_dict['train rmse ' + type_key].append(train_rmse)
                test_dict['test rmse ' + type_key].append(test_rmse)

            # change the ddicts into dataframes with rmse_table_tysi
            rmse_df_26, rmse_df_35, rmse_df_44 = \
                rmse_table_tysi(train_dict, test_dict, output_type)

        return rmse_df_26, rmse_df_35, rmse_df_44

    if output_type == 'site':
        # set up empty default dictionaries
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)

        for i in range(len(folds_dict_train)):

            # add a column to each dataframe (in each fold) temp that
            # classifies the defect site points into sub and int
            # do this step so groupby can be used
            folds_dict_train[i+1] = add_site_col(folds_dict_train[i+1])
            folds_dict_test[i+1] = add_site_col(folds_dict_test[i+1])

            # make each dict from rmse_predictor into a tuple of 2 tuples
            # (defect site (str), df of values associated with that sc type)
            # (('sub', df), ('int', df))
            # do this step so the dfs can be called.
            site_split_train = (tuple(folds_dict_train[i+1].groupby('Site2')))

            site_split_test = (tuple(folds_dict_test[i+1].groupby('Site2')))

            for site_idx in range(len(site_split_train)):

                y_train = site_split_train[site_idx][1]['dft_train']

                trainpred = site_split_train[site_idx][1]['pred_train']

                y_test = site_split_test[site_idx][1]['dft_test']
                testpred = site_split_test[site_idx][1]['pred_test']

                train_rmse, test_rmse = \
                    rmse(y_train, y_test, trainpred, testpred)

                site_key = site_split_train[site_idx][0]

                # append the train and test rmse to the ddict
                train_dict['train rmse ' + site_key].append(train_rmse)
                test_dict['test rmse ' + site_key].append(test_rmse)

            rmse_df_sub, rmse_df_int = \
                rmse_table_tysi(train_dict, test_dict, output_type)

        return rmse_df_sub, rmse_df_int

    else:
        print("value entered for output_type is not recognized. Use \
'none', 'type', or 'site'. ")

###############################################################################
###############################################################################


def preds_iterator(X, y, train_idx, test_idx, clf, Y_train, Y_test,
                   PRED_train, PRED_test, Y_tr_idx, Y_te_idx):
    '''
    Creates 4 arrays: Y_train, Y_test, PRED_train, PRED_test, made up of y
    train data y test data, y train predictions, and y test predictions
    respectively. For each iteration of train/test split, a new group of
    points is added to each of these arrays. Used in rfr_pp_predictor.

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
    # this func runs for n folds x n folds num of times. For each iteration:

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
    num of times). Used in rfr_pp_predictor.

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
    # this func runs for n folds x n folds num of times. For each iteration:

    # zip the train/test DFT points with their indexes into tuples (idx, DFT)
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


def pp_traintest_df(train_dict, test_dict, df):
    '''
    Takes in default dictionaries from zip_to_ddict of predicted values for
    the train and test sets. Returns dataframes (tables) that include the
    type/ab/impurity/site plus dft point/mean predicted value/std dev of
    predicted values. Used in rfr_pp_predictor.

    Inputs
        - train_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times).
        - test_dict: default dict. keys are DFT points, values are predicted
        points across folds of CV (repeated folds num of times)
        - df: pd df.

    Outputs
        - train_df_final: pd df. Data from the 'train set during CV. Columns:
        dft data, mean, std dev of predicted data. Type/ab/ impurity/defect
        of every predicted point.
        - test_df_final: pd df. Data from the 'test set' during CV. Columns:
        dft data, mean, std dev of predicted data. Type/ab/impurity/defect
        of every predicted point.
    '''

    # get the idx, DFT points, mean and std dev of prediction points
    # into separate lists

    idx_train, dft_train, mean_train, stddev_train = dict_sorter(train_dict)
    idx_test, dft_test, mean_test, stddev_test = dict_sorter(test_dict)

    train_dataframe = \
        pd.DataFrame(data={'index_train': idx_train, 'dft_train': dft_train,
                     'mean_train': mean_train, 'stddev_train': stddev_train})
    test_dataframe = \
        pd.DataFrame(data={'index_test': idx_test, 'dft_test': dft_test,
                     'mean_test': mean_test, 'stddev_test': stddev_test})

    train_df_final = join_pred_labels(train_dataframe, df)
    test_df_final = join_pred_labels(test_dataframe, df)

    return train_df_final, test_df_final


def dict_sorter(dic):
    '''
    Sorts default dictionaries which contain DFT vals as keys, and predicted
    vals from RFR cross validation as values into 3 lists. A list of keys,
    a list of the mean of values across the folds of cv, and a list of the std
    of the values across the folds of cv. Lists will be made into dataframes.
    Used in pp_traintest_df.

    Inputs:
        - dic: default dict. Keys are DFT values. Values are predicted values
        from cv during RFR.
    Outputs:
        - dft_list: list. keys from default dict.
        - mean_list: list. mean of the values associated with every key.
        - std_list list. std dev of the values associated with every key.
    '''
    # empty lists for each operation that will be performed on the values
    # in the train or test dictionaries
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

    return idx_list, dft_list, mean_list, std_list


def rfr_pp_predictor(df, o=0, d_start=5, num_trees=100, max_feat='auto',
                     max_depth=5, min_samp_leaf=2, min_samples_split=5,
                     folds=5, label_type=1, label_site=4):
    '''
    Wrapper func for the RFR model predictions for parity plot. Returns
    predictions and uncertainty for train and test points to be plotted.
    Uncertainty comes from n-folds of cross validation performed n-folds time.
    Tables are returned that have the actual dft points, the mean of the
    predicted points, and the std across the folds of cross validation. RMSE
    tables are returned that give a metric for how the model is predicting on
    a particular data set. The tables from this function can be used in
    parity_plot function.

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

    Outputs
        - train_df: pd dataframe. Contains mean/std dev of predicted values for
        the data points that were in the training set. Columns:
        type/ab/impurity/site/dft_train/mean_train/stddev_train.
        - test_df: pd dataframe. Contains mean/std dev of predicted values for
        the data points that were in the test set. Columns:
        type/ab/impurity/site/dft_test/mean_test/stddev_test.
    '''
    b = stratify_df(df, label_type, label_site)

    X, y = descriptors_outputs(df, d_start, o)

    skf_plot = StratifiedKFold(n_splits=folds, shuffle=True)

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

        for train_idx, test_idx in skf_plot.split(df, b):

            (Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx) = \
                preds_iterator(X, y, train_idx, test_idx, clf, Y_train, Y_test,
                               PRED_train, PRED_test, Y_tr_idx, Y_te_idx)

    train_dict, test_dict = \
        zip_to_ddict(Y_train, Y_test, PRED_train, PRED_test,
                     Y_tr_idx, Y_te_idx)

    train_df, test_df = \
        pp_traintest_df(train_dict, test_dict, df)

    return train_df, test_df


def plot_sorter(test_df, output_type):
    '''
    Sorts the test dataframe (test_df) from rfr_pp_predictor into separate
    dataframes by 'type' or 'site', depending on output_type selected. Used
    in parity_plot function.

    Inputs
        - test_df: pandas df. Contains mean/std dev of predicted values for
        the data points that were in the test set. Columns:
        type/ab/impurity/site/dft_test/mean_test/stddev_test.
        - output_type: str. Values must be 'type', 'site'.

    Outputs
        - testpred_: pandas df. dataframe of only one type of semiconductor,
        or one type of defect site. See notes.

    Notes: If output = 'type', returns 3 dfs (one for each sc type). If output
    = 'site', returns 2 dfs (one for each type of defect site (sub/int)).
    '''
    if output_type == 'type':
        idx_26 = []
        idx_35 = []
        idx_44 = []

        for idx in range(len(test_df)):
            if (test_df['Type'].loc[idx] == 'II-VI'):
                idx_26.append(idx)
            if (test_df['Type'].loc[idx] == 'III-V'):
                idx_35.append(idx)
            if (test_df['Type'].loc[idx] == 'IV-IV'):
                idx_44.append(idx)

        testpred_26 = test_df.loc[idx_26]
        testpred_35 = test_df.loc[idx_35]
        testpred_44 = test_df.loc[idx_44]

        return testpred_26, testpred_35, testpred_44

    if output_type == 'site':
        idx_sub = []
        idx_int = []

        test_df = add_site_col(test_df)

        for idx in range(len(test_df)):
            if (test_df['Site2'].loc[idx] == 'sub'):
                idx_sub.append(idx)
            elif (test_df['Site2'].loc[idx] == 'int'):
                idx_int.append(idx)

        testpred_sub = test_df.loc[idx_sub]
        testpred_int = test_df.loc[idx_int]

        return testpred_sub, testpred_int


def parity_plot(train_df, test_df, output_type, output_name):
    '''
    Plots the prediction tables from rfr_pp_predictor as a parity plot.
    In the parity plot the train and test points are both plotted. If 'none'
    is selected for output_type, the plot does not distinguish the test points.
    The plot can have the test points distinguished by semiconductor type
    or defect site by changing output_type to 'type' or 'site', respectively.

    Inputs
        - train_df: pd dataframe. Contains mean/std dev of predicted values for
        the data points that were in the training set. Columns:
        type/ab/impurity/site/dft_train/mean_train/stddev_train.
        - test_df: pd dataframe. Contains mean/std dev of predicted values for
        the data points that were in the test set. Columns:
        type/ab/impurity/site/dft_test/mean_test/stddev_test.
        - output_type: str. Values must be 'none', 'type', or 'site'.
        - output_name: str. Name of the output being plotted (for plot title)
        (ie dHA, (+3/+2), etc...)

    Outputs

    Notes: function automatically returns a parity plot
    '''
    # parity plot with uncertainties
    if output_type == 'none':
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(train_df['dft_train'], train_df['mean_train'],
                    yerr=train_df['stddev_train'], alpha=0.5, label='train',
                    color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(test_df['dft_test'], test_df['mean_test'],
                    yerr=test_df['stddev_test'], alpha=0.5, label='test',
                    color='red', zorder=3, fmt='o', markersize=4)
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
        return

    elif output_type == 'type':
        test26_df, test35_df, test44_df = plot_sorter(test_df, output_type)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(train_df['dft_train'], train_df['mean_train'],
                    yerr=train_df['stddev_train'], alpha=0.5, label='train',
                    color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(test26_df['dft_test'], test26_df['mean_test'],
                    yerr=test26_df['stddev_test'], alpha=0.5,
                    label='II-VI test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(test35_df['dft_test'], test35_df['mean_test'],
                    yerr=test35_df['stddev_test'], alpha=0.5,
                    label='III-V test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(test44_df['dft_test'], test44_df['mean_test'],
                    yerr=test44_df['stddev_test'], alpha=0.5,
                    label='IV-IV test', zorder=3, fmt='o', markersize=4)
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
        return

    elif output_type == 'site':
        testsub_df, testint_df = plot_sorter(test_df, output_type)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(train_df['dft_train'], train_df['mean_train'],
                    yerr=train_df['stddev_train'], alpha=0.5, label='train',
                    color='gray', zorder=3, fmt='o', markersize=4)
        ax.errorbar(testsub_df['dft_test'], testsub_df['mean_test'],
                    yerr=testsub_df['stddev_test'], alpha=0.5,
                    label='sub site test', zorder=3, fmt='o', markersize=4)
        ax.errorbar(testint_df['dft_test'], testint_df['mean_test'],
                    yerr=testint_df['stddev_test'], alpha=0.5,
                    label='int site test', zorder=3, fmt='o', markersize=4)

        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        lim = [np.min([ax.get_xlim(), ax.get_ylim()]),
               np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lim, lim, color='black', alpha=0.7)
        ax.set_title(output_name)
        ax.legend()
        return

    else:
        print("value entered for output_type is not recognized. Use \
'none', 'type', or 'site'. ")
        return
