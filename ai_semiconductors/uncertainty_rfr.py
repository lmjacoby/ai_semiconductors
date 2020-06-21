import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import statistics

from tqdm.notebook import tqdm


def uncertainty_rfr_qfr(df, X, Y, o, d_start=12, label_start=0, label_end=5,
                        true_y=True, max_depth=5, num_trees=1000,
                        min_samples_split=2, rs_split=130):
    '''
    This function calculates uncertainty for predicted values based on a
    RFR model trained on a specific output. It's based off the idea of a
    quantile regression forest. The model records ALL observed responses in
    each tree leaf in the forest (instead of recording the mean val of the
    response variable in each tree leaf). This allows confidence intervals to
    be calculated by analyzing the distribution of the response variables on
    the leaves. This is put into practice in python by fully expanding a tree
    so that each leaf has one val (ie min_samples_leaf=1).

    Inputs
        - df: pd df. Dataframe to train the model. Therefore the df must
        have X and Y (actual) values.
        - X: pd df. The data points you are trying to predict uncertainties
        on. ex: X_test or X of a new dataset (possibly with no Y)
        - Y: The actual values of the data points. ex: Y_test, if no actual
        vals put anything here, and make act='no'
        - label_start: int. start col index of where type/ab/m/site.
        - label_end: int. end col index of where type/ab/m/site.
        - true_y: binary. True if there are actual values associated with X,
        if no actual values (ie brand new points) then False (default:True)
        - o: int. Which output to train the model on (corresponds to column
        index in "output")
        - num_trees: int. Number of estimators (trees) to by used by the
        RFR in the random forest. Default:100.
        - max_feat: str. The number of features to consider when looking for
        the best split. Default: 'auto'
        - max_depth: int. The maximum depth of each tree in the forest.
        Keep this value low to mitigate overfitting. Default:5.
        - min_samples_split: int. The minimum number of samples required to
         split an internal node. Default: 5.
        - rs_split: int. value on which to start the random state generator.
        Ensures the bootstrapping in the RFR is consistent across runs.

    Outputs
        - err_df: pd df. contains idx of original datapoint, actual target
        value (if it exists), 95% confidence inverval and lower limit/upper
        limit of predictions, mean prediction, std. dev of predicions.

    '''
    # This section of code is all for training the model
    descriptors = df.columns[d_start:]
    # output = df.columns[o]

    P = df[descriptors]
    s = df[df.columns[label_start:label_end]]

    X_train, X_test, y_train, y_test = \
        train_test_split(P, s, test_size=0.22, random_state=rs_split,
                         stratify=s[['Type', 'Site']])

    clf = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth,
                                min_samples_leaf=1,
                                min_samples_split=min_samples_split,
                                random_state=130, n_jobs=-1)

    # train the RFR model on the descriptors and specific output
    clf.fit(X_train[descriptors], y_train[y_train.columns[o]])

    # make predictions on the NEVER BEFORE SEEN points
    err_down = []
    err_up = []
    err_mean = []
    err_stddev = []
    X_arr = np.array(X)
    for x in tqdm(np.arange(len(X_arr))):
        preds = []
        for pred in clf.estimators_:
            preds.append(pred.predict(X_arr[x].reshape(1, -1))[0])

        err_down.append(np.quantile(preds, 0.025))
        err_up.append(np.quantile(preds, 0.975))
        err_mean.append(np.mean(preds))
        err_stddev.append(statistics.pstdev(preds))

    if true_y is True:
        Y_arr = np.array(Y)
    elif true_y is False:
        Y_arr = np.zeros(X_arr.shape[0])

    truth = Y_arr

    index = list(X.index)

    d = {'index': index, 'actual': truth, 'err_down': err_down,
         'err_up': err_up, 'predicted': err_mean, 'std_dev': err_stddev}

    err_df = pd.DataFrame(data=d)

    err_df['err_interval'] = err_df['err_up'] - err_df['err_down']
    err_df.set_index('index', inplace=True)

    return err_df


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
        - y_train: np array. output values of training data set.
    '''
    X_train = X.iloc[list(train_idx)]
    y_train = y.iloc[list(train_idx)]

    return X_train, y_train


def predict_append(clf, N_arr, n, preds):
    '''
    Appends prediction from the RFR model for every point in that fold to a
    list. List will be added to a np array.

    Inputs
        - clf: RandomForestRegressor from sklearn
        - N_arr: np array. X descriptors made into an array.
        - n: int. index of N_array to predict on.
        - preds: list. list to append prediction values to.
    Outputs
        - pred: list. list with appended values. Will be added to an array
        as one one at the position k-fold (ie first fold is first row).
    '''
    pred = clf.predict(N_arr[n].reshape(1, -1))[0]
    preds.append(pred)

    return preds


def dft_points(true_y, Y, N_arr):
    '''
    makes or pulls values to be added to final df err_df.

    Inputs
        - true_y: bool. If True, there are true y-values to append. if False,
        the array is 0's
        - Y: if true_y is True, this should be a pd df. if true_y is False,
        put anything here.
        - N_arr: np array. X descriptors made into an array.
    Outputs
        - Y_arr: np array. either true Y values or 0's to be added to
        the columns 'true_y' in err_df
    '''
    if true_y is True:
        Y_arr = np.array(Y)
    elif true_y is False:
        Y_arr = np.zeros(N_arr.shape[0])

    return Y_arr


def uncert_table(N, X, type_col, ab_col, site_col, imp_col, Y_arr,
                 pred_df_desc):
    '''
    Makes a nice output table of the mean value and std dev per point in X,
    and std dev. Also includes index of n, type, sc, site, impurity and
    true_y(if applicable).

    Inputs:
        - N: pd df. X descriptors, formed using x_start.
        - X: pd df. The data points you are trying to predict uncertainties
        on. ex: X_test or X of a new dataset (possibly with no Y)
        - type_col: int. column index of 'Type' column in X.
        - ab_col: int. column index of 'AB' column in X.
        - site_col: int. column index of 'Site' column in X.
        - imp_col: int. column index of 'Impurity/M' column in X.
        - Y_arr: np array. either true Y values or 0's to be added to
        the columns 'true_y' in err_df
        - pred_df_desc: pd df. call describe on the np array that has all the
        predicted values across the folds to get mean and std dev.

    Outputs:
        - err_df: pd df. A dataframe with the type, sc, site, impurity, mean
        and std dev across k-folds for every point in X.

    '''

    d = {'index': list(N.index), 'Type': list(X[X.columns[type_col]]),
         'AB': list(X[X.columns[ab_col]]),
         'Site': list(X[X.columns[site_col]]),
         'Impurity': list(X[X.columns[imp_col]]), 'true val': Y_arr,
         'mean': pred_df_desc.T['mean'], 'std': pred_df_desc.T['std']}

    err_df = pd.DataFrame(data=d)
    err_df.set_index('index', inplace=True)

    return err_df


def uncertainty_rfr_cv(df, X, Y, o, d_start=5, x_start=4, true_y=False,
                       max_depth=5, num_trees=100, min_samp_leaf=2,
                       min_samples_split=2, max_feat='auto', folds=5,
                       type_col=0, ab_col=1, site_col=2, imp_col=3):
    '''
    This function calculates uncertainty for predicted values based on cross
    validation of a RFR model. A model is fit on some part of the data (all
    data but the k-fold), and then predicts a val for each of a set of unknown
    points. It does this k-fold times, and then the mean of the k-fold
    predictions and standard deviation of the k-fold predictions is calculated
    for each of the unknown poitns. The unknown points and the training data
    must have the same descriptors.

    Inputs
        - df: pd df. Dataframe to train the model. Therefore the df must
        have X and Y (actual) values.
        - X: pd df. The data points you are trying to predict uncertainties
        on. ex: X_test or X of a new dataset (possibly with no Y)
        - Y: pd df. The true values of the data points. ex: Y_test, if no
        actual vals put anything here, and make true_y= False
        - o: int. Which output to train the model on (corresponds to column
        index in "output")
        - d_start: int. column index that the descriptors columns start in df.
        In the input df, the descriptors must start at some column at index
        df_start to the last column in the dataframe. Default: 5.
        - x_start: int. column index that the descriptor columns start in X.
        Default:5
        - true_y: binary. True if there are actual values associated with X,
        if no actual values (ie brand new points) then False (default:True)
        - num_trees: int. Number of estimators (trees) to by used by the
        RFR in the random forest. Default:100.
        - max_feat: str. The number of features to consider when looking for
        the best split. Default: 'auto'
        - min_samp_leaf: int. The minimum number of samples required to be at
         a leaf node. Deafult: 2.
        - max_depth: int. The maximum depth of each tree in the forest.
        Keep this value low to mitigate overfitting. Default:5.
        - min_samples_split: int. The minimum number of samples required to
         split an internal node. Default: 5.
        - folds: int. Number of folds to to split the data in cross validation.
        Default: 5.
        - type_col: int. column index of 'Type' column in X.
        - ab_col: int. column index of 'AB' column in X.
        - site_col: int. column index of 'Site' column in X.
        - imp_col: int. column index of 'Impurity/M' column in X.

    Outputs
        - pred_df: pd df. A dataframe that contains all the values predicted
        from the model (all the folds x all the points).
        - err_df: pd df. A dataframe with the type, sc, site, impurity, mean
        and std dev across k-folds for every point in X.
    '''
    descriptors, output = descriptors_outputs(df, d_start, o)

    kf = KFold(n_splits=folds, shuffle=True, random_state=130)

    clf = RandomForestRegressor(n_estimators=num_trees,
                                max_features=max_feat, max_depth=max_depth,
                                min_samples_leaf=min_samp_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1, random_state=130)

    N = X[X.columns[x_start:]]

    # shape is folds rows x (num of data points predicting) columns
    preds_all = np.zeros((folds, N.shape[0]))

    count = -1

    for train_idx, test_idx in kf.split(descriptors, output):

        X_train, y_train = traintest(descriptors, output, train_idx, test_idx)

        # train the RFR model on the descriptors and specific output
        clf.fit(X_train, y_train)

        count += 1

        N_arr = np.array(N)
        preds = []
        for n in tqdm(np.arange(len(N_arr))):
            preds = predict_append(clf, N_arr, n, preds)
            # pred = clf.predict(N_arr[n].reshape(1,-1))[0]
            # preds.append(pred)

        preds_all[count] = preds

    Y_arr = dft_points(true_y, Y, N_arr)

    pred_df = pd.DataFrame(data=preds_all)
    pred_df_desc = pred_df.describe()

    err_df = uncert_table(N, X, type_col, ab_col, site_col, imp_col, Y_arr,
                          pred_df_desc)

    return pred_df, err_df


def largest_uncertainty(df, num_vals, column):
    '''
    Takes in a dataframe from uncertainty_calc func and returns a dataframe
    of the n largest uncertainties ordered by a particular column

    Inputs
        - df: pd df. dataframe from ucertainity_calc
        - num_vals: int. number of largest vals to return
        - column: str. column name in the df to sort largest by
        (ie 'std_dev' or 'err_interval')
    Outputs
        - df_largest: pd df. dataframe of n largest values sorted by column
        - idx: list. a list of the index values of the n largest uncertainties
    '''
    df_largest = df.nlargest(num_vals, column)
    idx = list(df_largest.index)

    return df_largest, idx
