import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas.api.types as ptypes

import uncertainty_rfr

df_test = pd.read_csv('./xiaofeng_lasso/unittest_dummy.csv', nrows=5)
X_test, y_test = uncertainty_rfr.descriptors_outputs(df_test, d_start=5,
                                                     o=0)


def test_uncertainty_rfr_qfr():
    '''
    Test function for uncertainty_rfr_qfr. Checks values in actual are 0 when
    true_y = False, and that the output df has the correct number of rows.
    '''
    df_test = pd.read_csv('./xiaofeng_lasso/unittest_dummy.csv')
    X = df_test.iloc[range(3)]

    err_df_test = \
        uncertainty_rfr.uncertainty_rfr_qfr(df_test, X[X.columns[5:]],
                                            Y='none', true_y=False, o=0,
                                            d_start=5)

    assert err_df_test['actual'][0] == err_df_test['actual'][1], \
        'with true_y = False, all values in "actual" should be equal (0.0)'
    assert len(err_df_test) == len(X), \
        'length of predicting df should equal length of output df'


def test_descriptors_outputs():
    '''
    Test function for descriptors_outputs. Checks the shape of X, and checks
    that the correct type of value (numeric) is in the columns.
    '''
    X_test, y_test = uncertainty_rfr.descriptors_outputs(df_test, d_start=5,
                                                         o=0)

    assert X_test.shape[1] == 5, \
        'array shape is incorrect. should be ({}, 7), got ({}, {})'\
        .format(X_test.shape[0], X_test.shape[0], X_test.shape[1])

    assert all(ptypes.is_numeric_dtype(X_test[col]) for col in
               list(X_test[X_test.columns[:]])), \
        'data type in columns is of incorrect type, must be numeric'

    assert ptypes.is_numeric_dtype(y_test), \
        'data type in columns is of incorrect type, must be numeric'


def test_traintest():
    '''
    Test function for traintest. Checks that the length of X_train and
    y_train are the same.
    '''
    train_idx_test = np.array([0, 1, 2])
    test_idx_test = np.array([3, 4])

    X_train_test, y_train_test = \
        uncertainty_rfr.traintest(X_test, y_test, train_idx_test,
                                  test_idx_test)

    assert X_train_test.shape[0] == y_train_test.shape[0], \
        'X_train and y_train datapoints do not have the same num of values'


def test_predict_append():
    '''
    Test function for predict_append. Checks that the func appends one value
    at a time, and that the output is a list.
    '''
    df_test2 = df_test[df_test.columns[:7]]
    X_test, y_test = uncertainty_rfr.descriptors_outputs(df_test2, d_start=5,
                                                         o=0)
    clf_test = RandomForestRegressor(random_state=130)
    clf_test.fit(X_test, y_test)
    N_arr_test = np.array([[3.98069889, 0.38048415],
                          [-0.78001682, 0.20058657]])
    n_test = 0
    preds_test = []

    preds_test = uncertainty_rfr.predict_append(clf_test, N_arr_test, n_test,
                                                preds_test)

    assert len(preds_test) == 1, \
        'preds_test needs to be length 1. Got {}'.format(len(preds_test))

    assert isinstance(preds_test, list), \
        'preds_test needs to be a list, got {}'.format(type(preds_test))


def test_dft_points():
    '''
    Test functino for dft_points. Checks that when true_y = True, the output
    array is equal to Y_test, adn when true_y = False the output arry is the
    same length as N_arr_test.
    '''
    Y_test = [3, 5]
    N_arr_test = np.array([[3.98069889, 0.38048415],
                          [-0.78001682, 0.20058657]])

    Y_arr_test = uncertainty_rfr.dft_points(True, Y_test, N_arr_test)
    Y_arr_test2 = uncertainty_rfr.dft_points(False, Y_test, N_arr_test)

    assert Y_arr_test[0] == Y_test[0], \
        'Y_arr_test got unexpected result. Expected np.array([3,5]), got{}'.\
        format(Y_arr_test)
    assert len(Y_arr_test2) == N_arr_test.shape[0], \
        'length of Y_arr_test2 should be equal to the number of rows of \
         N_arr_test. Got Y_arr: {}, N_arr {}'.\
        format(len(Y_arr_test2), N_arr_test.shape[0])


def test_uncert_table():
    '''
    Test function for uncert_table. Checks that the columns in the df are in
    the correct place, the length of the output dataframe the correct
    length, and that the last three columns in the output df are numeric.
    '''
    N_test = df_test[df_test.columns[5:]].iloc[[0, 1]]
    X = df_test.iloc[[0, 1]]
    Y_arr_test = np.array([3, 5])
    pred_desc_test = pd.DataFrame(data={'mean': [1, 2], 'std': [3, 4]}).T

    err_df = uncertainty_rfr.uncert_table(N_test, X, 1, 2, 3, 4,
                                          Y_arr_test, pred_desc_test)

    assert err_df.columns[0] == 'Type', \
        'first column got unexpected value {}, should be Type'.\
        format(err_df.columns[0])
    assert len(err_df) == len(X), \
        'arrays must all be the same length'
    assert all(ptypes.is_numeric_dtype(err_df[col]) for col in
               list(err_df[err_df.columns[4:]])), \
        'columns "true val", "mean", and "std" are of wrong type, should be\
         numeric values.'


def test_uncertainty_rfr_cv():
    '''
    Test function for undertainty_rfr_cv. Checks that the prediction df has
    as many rows as folds in cv. In the output df it checks that "true val"
    values are 0 when true_y = False, and checks that values in "AB" are of
    type string.
    '''
    X = df_test.iloc[[0, 1]]
    Y = 'none'
    d_start, x_start = 5, 5
    o = 0
    folds_test = 2

    pred_df_test, err_df_test = \
        uncertainty_rfr.uncertainty_rfr_cv(df_test, X, Y, o, d_start, x_start,
                                           folds=folds_test)

    assert pred_df_test.shape[0] == folds_test, \
        'Number of row in pred_df_test array should equal number of folds, \
        expected {}, got {}'.format((folds_test, pred_df_test.shape[0]))
    assert err_df_test[err_df_test.columns[4]][0] == 0.0, \
        'Expected 0.0 in "true val" with true_y set to false, instead got a \
        different val'
    assert isinstance(err_df_test['AB'][1], str), \
        'Expected string in column "AB", got {}'.format(type(
         err_df_test['AB'][1]))


def test_largest_uncertainty():
    '''
    test function for largest_uncertainty. checks that that length of the
    df is equal to the num of values it was asked to return, and that the
    output idx are a list.
    '''
    df = pd.DataFrame(data={'err_int': [1, 2, 3], 'std_dev': [4, 5, 6]})
    num_vals = 2

    larg, idx = uncertainty_rfr.largest_uncertainty(df, num_vals, 'std_dev')

    assert len(larg) == num_vals, \
        'number of rows in the output df should equal the number of values\
        the func called to return'
    assert isinstance(idx, list), \
        'expected idx to be list, got {}'.format(type(idx))
