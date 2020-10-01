#!/usr/bin/env python
# coding: utf-8

# In[1]:
from contextlib import contextmanager
from io import StringIO

import sys
sys.path.append("../")
import rfr_model   # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import pandas.api.types as ptypes  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from collections import defaultdict  # noqa: E402


df_test = pd.read_csv('./unittest_dummy.csv', nrows=5)
X_test, y_test = rfr_model.descriptors_outputs(df_test, d_start=5, o=0)


def test_stratify_df():
    '''
    '''
    b_test = rfr_model.stratify_df(df_test, label_type=1, label_site=4)

    assert b_test.shape[1] == 1, \
        'array shape is incorrect. should be ({}, 1), got ({}, {})'\
        .format(b_test.shape[0], b_test.shape[0], b_test.shape[1])

    assert isinstance(b_test, np.ndarray), \
        'output type is incorrect, should be of type np array.'


def test_descriptors_outputs():
    ''
    ''
    X_test, y_test = rfr_model.descriptors_outputs(df_test, d_start=5, o=0)

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
    '''
    train_idx_test = np.array([0, 1, 2])
    test_idx_test = np.array([3, 4])

    X_train_test, X_test_test, y_train_test, _ = \
        rfr_model.traintest(X_test, y_test, train_idx_test, test_idx_test)

    assert len(X_train_test) >= len(X_test_test), \
        'more testing data points than training datapoints, make testsize \
    smaller'

    assert X_train_test.shape[0] == y_train_test.shape[0], \
        'X_train and y_train datapoints do not have the same num of values'


def test_fit_predict():
    '''
    '''
    X_train_test = X_test.iloc[[0, 1, 2]]
    X_test_test = X_test.iloc[[3, 4]]
    y_train_test = y_test.iloc[[0, 1, 2]]
    clf_test = RandomForestRegressor(random_state=130)

    trainpred_test, testpred_test = \
        rfr_model.fit_predict(X_train_test, y_train_test,
                              X_test_test, clf_test)

    assert len(trainpred_test) >= len(testpred_test), \
        'more testing data points than training datapoints, make testsize \
    smaller'

    assert isinstance(testpred_test[0], np.float), \
        'predicted value of the wrong type, must be np.float'


def test_rmse():
    '''
    '''
    y_train_test = y_test.iloc[[0, 1, 2]]
    y_test_test = y_test.iloc[[3, 4]]
    trainpred_test = np.array([6.85622831, 6.95560622, 7.03754682])
    testpred_test = np.array([6.96804053, 6.99679452])

    train_rmse_test, test_rmse_test = \
        rfr_model.rmse(y_train_test, y_test_test, trainpred_test,
                       testpred_test)

    assert isinstance(train_rmse_test, np.float), \
        'predicted train rmse value of the wrong type, must be np.float'

    assert test_rmse_test > 0, \
        'predicted test rmse value is below 0, must be above zero'


def test_rmse_list():
    '''
    '''
    train_list_test = []
    test_list_test = []

    train_rmse_test = 2.1035121088348996
    test_rmse_test = 3.4289289813688395

    train_list_test, test_list_test = \
        rfr_model.rmse_list(train_list_test, test_list_test, train_rmse_test,
                            test_rmse_test)

    assert isinstance(train_list_test, list), \
        'output type is incorrect. should be list, got {}'\
        .format(type(train_list_test))

    assert len(test_list_test) > 0, \
        'output list is empty, values are not appending'


def test_rmse_table_ms():

    train_list_test = [1, 2, 3]
    test_list_test = [4, 5, 6]

    rmse_df = rfr_model.rmse_table_ms(train_list_test, test_list_test)

    assert isinstance(rmse_df.loc[3][0], str), \
        'mean/stddev val is of wrong type, should be string'


def test_rmse_table_tysi():
    train_dict_ty = \
        defaultdict(list,
                    {'train rmse II-VI': [0.03531],
                     'train rmse III-V': [0.05715],
                     'train rmse IV-IV': [0.01928]})
    test_dict_ty = \
        defaultdict(list,
                    {'test rmse II-VI': [0.035315],
                     'test rmse III-V': [0.057154],
                     'test rmse IV-IV': [0.019287]})
    train_dict_si = \
        defaultdict(list,
                    {'train rmse sub': [0.01841159],
                     'train rmse int': [0.04063153]})
    test_dict_si = \
        defaultdict(list,
                    {'test rmse sub': [0.018411594],
                     'test rmse int': [0.040631534]})

    df_26, _, _ = rfr_model.rmse_table_tysi(train_dict_ty, test_dict_ty,
                                            'type')
    _, df_int = rfr_model.rmse_table_tysi(train_dict_si, test_dict_si, 'site')

    assert df_26.columns[0] == 'train rmse II-VI', \
        'incorrect column in type II-VI dataframe, should be train rmse II-VI'

    assert df_26.shape[0] == 2, \
        'length of df is incorrect. Should be 2, instead got {}'\
        .format(df_26.shape[0])

    assert df_int.columns[1] == 'test rmse int', \
        'incorrect column in site int dataframe, should be train rmse int'

    assert isinstance(df_int.iloc[1][0], str), \
        'last row of df should be of type str'


def test_join_pred_labels():

    df_pred_test = pd.DataFrame({'index': [1, 4, 2],
                                 'dft_test': [6.5, 3.4, 4],
                                 'pred_test': [6.2, 3.0, 4.2]})

    final_df_test = rfr_model.join_pred_labels(df_pred_test, df_test)

    assert final_df_test.columns[0] == 'Type', \
        'incorrect first column, expected Type, got {}'\
        .format(final_df_test.columns[0])

    assert len(final_df_test) == len(df_pred_test), \
        'length of returned dataframe ({}) should be the same as length of \
    prediction dataframe ({}), but is not'\
        .format(len(final_df_test), len(df_pred_test))


def test_add_site_col():

    dict_df_test = rfr_model.add_site_col(df_test)

    assert dict_df_test.columns[-1] == 'Site2', \
        'New column was not appended in the correct position.'

    assert dict_df_test['Site2'][0] == 'int', \
        'Incorrect mapping from Site column. Expected int, got {}'\
        .format(dict_df_test['Site2'][0])


df_test_long = pd.read_csv('./unittest_dummy.csv', nrows=30)
folds_test = 2


def test_rfr_predictor():

    test_folds_dict_train, test_folds_dict_test = \
        rfr_model.rfr_predictor(df_test_long, folds=folds_test)

    assert len(test_folds_dict_train) == folds_test, \
        'dictionary of dataframes is not the right length, expected 2, \
    got {}'.format(len(test_folds_dict_train))

    assert len(test_folds_dict_train) == len(test_folds_dict_test), \
        'length of train and test dictionary of dataframes should be the same.\
    Got length train = {}, and length test = {}'\
        .format(len(test_folds_dict_train), len(test_folds_dict_test))

    assert isinstance(test_folds_dict_train[2]['Site'][1], str), \
        'value in site column is of the wrong type, should be str.'


def test_rmse_calculator():

    test_dtrain, test_dtest = \
        rfr_model.rfr_predictor(df_test_long, folds=folds_test)

    test_rmse_df = rfr_model.rmse_calculator(test_dtrain, test_dtest, 'none')

    assert test_rmse_df.columns[0] == 'train rmse', \
        'incorrect column name for first column. expected train rmse, got {}'\
        .format(test_rmse_df.columns[0])

    assert len(test_rmse_df) == folds_test + 1, \
        'rmse dataframe length is incorrect, should be 1 more than number of\
    folds'

    _, test_rmse_df35, test_rmse_df44 = \
        rfr_model.rmse_calculator(test_dtrain, test_dtest, 'type')

    assert test_rmse_df35.columns[1] == 'test rmse III-V', \
        'incorrect column name for second column. expected test rmse III-V,\
    got {}'.format(test_rmse_df35.columns[1])

    assert len(test_rmse_df35) == len(test_rmse_df44), \
        'rmse dataframes should be the same length, instead got III-V = {} \
    and IV-IV = {}'.format(len(test_rmse_df35), len(test_rmse_df44))

    test_rmse_dfsub, test_rmse_dfint = \
        rfr_model.rmse_calculator(test_dtrain, test_dtest, 'site')

    assert test_rmse_dfsub.columns[1] == 'test rmse sub', \
        'incorrect column name for second column. expected test rmse sub,\
    got {}'.format(test_rmse_dfsub.columns[1])

    assert len(test_rmse_dfsub) == len(test_rmse_dfint), \
        'rmse dataframes should be the same length, instead got sub = {} \
    and int = {}'.format(len(test_rmse_dfsub), len(test_rmse_dfint))


X, y = rfr_model.descriptors_outputs(df_test, d_start=5, o=0)

train_idx, test_idx = list(df_test.index)[:3], list(df_test.index)[3:]

clf = RandomForestRegressor(n_estimators=100, max_features='auto',
                            max_depth=5, min_samples_leaf=2,
                            min_samples_split=5, n_jobs=-1, random_state=130)

Y_tr, Y_te, PRED_tr, PRED_te, Y_tridx, Y_teidx = \
    (np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0),
     np.empty(0))


def test_preds_iterator():

    Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx = \
        rfr_model.preds_iterator(X, y, train_idx, test_idx, clf, Y_tr, Y_te,
                                 PRED_tr, PRED_te, Y_tridx, Y_teidx)

    assert len(Y_train) == len(PRED_train), \
        'Y_train and PRED_train should be the same length, instead got {} and\
    {}'.format(len(Y_train), len(PRED_train))

    assert y[3] == Y_test[0], \
        'y[3] should be the same as the Y_test[0], instead got {} and {}'\
        .format(y[3], Y_test[0])

    assert isinstance(PRED_test, list), \
        'PRED_train should be of type list, got {}'.format(type(PRED_train))

    assert isinstance(Y_tr_idx[1], float), \
        'Y_tr_idx[1] should be of type float, got {}'.format(type(Y_tr_idx[1]))


def test_zip_to_ddict():

    Y_train, Y_test, PRED_train, PRED_test, Y_tr_idx, Y_te_idx = \
        rfr_model.preds_iterator(X, y, train_idx, test_idx, clf, Y_tr, Y_te,
                                 PRED_tr, PRED_te, Y_tridx, Y_teidx)

    traind, testd = rfr_model.zip_to_ddict(Y_train, Y_test, PRED_train,
                                           PRED_test, Y_tr_idx, Y_te_idx)

    assert len(traind) > len(testd), \
        'train dictionary should be longer than test dictionaries, instead \
    got {} and {}'.format(len(traind), len(testd))

    assert isinstance(list(traind)[0], tuple), \
        'keys in traind dictionary should be of type tuple, instead got {}'\
        .format(type(list(traind)[0]))

    assert isinstance(list(testd.values())[1], list), \
        'values in testd dictionary should be of type list, instead got {}'\
        .format(type(list(testd.values())[1]))


def test_pp_traintest_df():

    traind = defaultdict(list, {(0.0, 6.769985632999999): [6.7, 6.8],
                                (1.0, 6.979431759): [6.8, 7.0],
                                (2.0, 7.095054806): [7.9, 8.2]})
    testd = defaultdict(list, {(3.0, 6.769985632999999): [8.9, 6.5],
                               (4.0, 6.979431759): [4.6, 5.0]})

    pp_trdf, pp_tedf = rfr_model.pp_traintest_df(traind, testd, df_test)

    assert pp_trdf.columns[1] == 'AB', \
        'second column in dataframe should be AB, got {}'\
        .format(pp_trdf.columns[1])

    assert len(pp_tedf) == len(testd), \
        'test dataframe should be same length as test dictionary, instead got\
    {} and {}'.format(len(pp_tedf), len(testd))


def test_dict_sorter():
    dic_test = {(2.0, 'a'): [1, 2, 3], (4.0, 'b'): [1, 2, 3]}

    _, dft_test, mean_test, std_test = rfr_model.dict_sorter(dic_test)

    assert isinstance(dft_test[0], str), \
        'value in dft list is of the wrong type, should be str.'

    assert len(mean_test) == 2, \
        'incorrect num of values appended to list, expected 2, got {}'\
        .format(len(mean_test))

    assert std_test[1] == 1, \
        'standard deviation calculated incorrectly. expected 1, got {}'\
        .format(std_test[1])


def test_rfr_pp_predictor():

    pp_trdf, pp_tedf = \
        rfr_model.rfr_pp_predictor(df_test_long, folds=folds_test)

    assert len(pp_trdf) == len(pp_tedf), \
        'length of train and test dataframes should be the same. Got length\
    train = {}, and length test = {}'.format(len(pp_trdf), len(pp_tedf))

    assert len(pp_tedf) == len(df_test_long), \
        'length of test dataframe and input dataframe should be the same. \
    Got length test = {}, and length input = {}'\
        .format(len(pp_tedf), len(df_test_long))

    assert isinstance(pp_trdf['Site'][1], str), \
        'value in site column is of the wrong type, should be str.'


def test_plot_sorter():

    pp_trdf, pp_tedf = \
        rfr_model.rfr_pp_predictor(df_test_long, folds=folds_test)

    te26, te35, te44 = rfr_model.plot_sorter(pp_tedf, 'type')

    assert te26.columns[4] == 'dft_test', \
        "test df fifth column is incorrect, should be 'dft_train'"

    assert len(te35) < len(te44), \
        'dataframes are not the correct length'

    assert isinstance(te44['Site'].iloc[5], str), \
        'value in site column is of the wrong type, should be str.'

    tesub, teint = rfr_model.plot_sorter(pp_tedf, 'site')

    assert len(tesub) < len(teint), \
        'length of integer dataframe should be longer than substitutional\
    dataframe. Instead got {} and {}'.format(len(tesub), len(teint))

    assert tesub.columns[-1] == 'Site2', \
        'incorrect last column in dataframe, should be Site2, instead got {}'\
        .format(tesub.columns[-1])


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_parity_plot():

    pp_trdf, pp_tedf = \
        rfr_model.rfr_pp_predictor(df_test_long, folds=folds_test)

    with captured_output() as (out, err):
        rfr_model.parity_plot(pp_trdf, pp_tedf, output_type='foo',
                              output_name='foo')

    output = out.getvalue().strip()
    assert isinstance(output, str), \
        'using a unrecognized call for output_type should return a print \
        statement, instead got {}'.format(type(output))
