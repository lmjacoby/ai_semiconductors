import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import pandas.api.types as ptypes

from ai_semiconductors import RFR_LASSO

df_test = pd.read_csv('./unittest_dummy.csv', nrows=5)
X_test, y_test = RFR_LASSO.descriptors_outputs(df_test, d_start=5, o=0)


def test_stratify_df():
    '''
    '''
    b_test = RFR_LASSO.stratify_df(df_test, label_type=1, label_site=4)

    assert b_test.shape[1] == 1, \
        'array shape is incorrect. should be ({}, 1), got ({}, {})'\
        .format(b_test.shape[0], b_test.shape[0], b_test.shape[1])
    assert isinstance(b_test, np.ndarray), \
        'output type is incorrect, should be of type np array.'


def test_descriptors_outputs():
    ''
    ''
    X_test, y_test = RFR_LASSO.descriptors_outputs(df_test, d_start=5, o=0)

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
        RFR_LASSO.traintest(X_test, y_test, train_idx_test, test_idx_test)

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
        RFR_LASSO.fit_predict(X_train_test, y_train_test,
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
        RFR_LASSO.rmse(y_train_test, y_test_test, trainpred_test,
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
        RFR_LASSO.rmse_list(train_list_test, test_list_test, train_rmse_test,
                            test_rmse_test)

    assert isinstance(train_list_test, list), \
        'output type is incorrect. should be list, got {}'\
        .format(type(train_list_test))
    assert len(test_list_test) > 0, \
        'output list is empty, values are not appending'


def test_rmse_total():
    train_list_test = []
    test_list_test = []
    train_idx_test = np.array([0, 1, 2])
    test_idx_test = np.array([3, 4])
    clf_test = RandomForestRegressor(random_state=130)

    # for train_idx_test, test_idx_test in skf.split(df_test, b_test):
    train_list_test, test_list_test, X_train_test, y_train_test = \
        RFR_LASSO.rmse_total(df_test, X_test, y_test, train_idx_test,
                             test_idx_test, clf_test, train_list_test,
                             test_list_test)

    assert isinstance(train_list_test, list), \
        'output type is incorrect. should be list, got {}'\
        .format(type(train_list_test))
    assert len(test_list_test) > 0, \
        'output list is empty, values are not appending'
    assert X_train_test.shape[0] == y_train_test.shape[0], \
        'X_train and y_train datapoints do not have the same num of values'


def test_rmse_table_ms():

    train_list_test = [1, 2, 3]
    test_list_test = [4, 5, 6]

    rmse_df = RFR_LASSO.rmse_table_ms(train_list_test, test_list_test)

    assert isinstance(rmse_df.loc[3][0], str), \
        'mean/stddev val is of wrong type, should be string'

############################################


df2_test = pd.read_csv('./unittest_dummy.csv')
X2_test, y2_test = RFR_LASSO.descriptors_outputs(df2_test, d_start=5, o=0)


def test_df_tysi():
    train_idx_test = np.array(range(30))
    test_idx_test = np.array(range(30, 40))

    train_idx_26_test, _, _, test_idx_26_test, _, _ = \
        RFR_LASSO.df_tysi(df2_test, train_idx_test, test_idx_test, 'type')

    train_idx_sub_test, _, test_idx_sub_test, _ = \
        RFR_LASSO.df_tysi(df2_test, train_idx_test, test_idx_test, 'site')

    assert len(train_idx_26_test) > 0, \
        'list of training indexes of sc type II-VI is empty, choose a larger \
        set or different points'
    assert df2_test.loc[test_idx_26_test[0]]['Type'] == 'II-VI', \
        'sc type does not match the list it was put into'
    assert len(train_idx_sub_test) > 0, \
        'list of training indexes of substitution defect sites is empty, \
        choose a larger set or different points'
    assert df2_test.loc[test_idx_sub_test[1]]['Site'] == 'M_A', \
        'defect site is not in group substitutional, wrong list'


def test_traintest_df_tysi():
    tr26, tr35, tr44, te26, te35, te44 = \
        ([0, 1, 3, 6, 8, 10, 11, 13, 16, 18, 20, 21, 23, 26, 28],
         [5, 9, 15, 19, 25, 29], [2, 4, 7, 12, 14, 17, 22, 24, 27],
         [30, 31, 33, 36, 38], [35, 39], [32, 34, 37])

    trsub, trint, tesub, teint = \
        ([1, 2, 11, 12, 21, 22],
         [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 6, 17, 18, 19, 20, 23, 24,
          25, 26, 27, 28, 29], [31, 32], [30, 33, 34, 35, 36, 37, 38, 39])

    p_type = RFR_LASSO.traintest_df_tysi(X2_test, y2_test, 'type', tr26,
                                         tr35, tr44, te26, te35, te44)

    p_site = RFR_LASSO.traintest_df_tysi(X2_test, y2_test, 'site',
                                         trsub, trint, tesub, teint)

    assert len(p_type) == 12, \
        'xy type dictionary has wrong num of values. expected 12, got{}'.\
        format(len(p_type))
    assert len(p_site) == 8, \
        'xy site dictionary has wrong num of values. expected 8, got{}'.\
        format(len(p_site))


def test_make_xy_list():
    tr26, tr35, tr44, te26, te35, te44 = \
        ([0, 1, 3, 6, 8, 10, 11, 13, 16, 18, 20, 21, 23, 26, 28],
         [5, 9, 15, 19, 25, 29], [2, 4, 7, 12, 14, 17, 22, 24, 27],
         [30, 31, 33, 36, 38], [35, 39], [32, 34, 37])

    trsub, trint, tesub, teint = \
        ([1, 2, 11, 12, 21, 22],
         [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 6, 17, 18, 19, 20, 23, 24,
          25, 26, 27, 28, 29], [31, 32], [30, 33, 34, 35, 36, 37, 38, 39])

    p_type = RFR_LASSO.traintest_df_tysi(X2_test, y2_test, 'type', tr26,
                                         tr35, tr44, te26, te35, te44)
    p_site = RFR_LASSO.traintest_df_tysi(X2_test, y2_test, 'site',
                                         trsub, trint, tesub, teint)

    xy_list_type = RFR_LASSO.make_xy_list(p_type, 'type')
    xy_list_site = RFR_LASSO.make_xy_list(p_site, 'site')

    assert len(xy_list_type) == len(p_type)/4, \
        'xy list type does not have the correct num of tuples'
    assert len(xy_list_site) == len(p_site)/4, \
        'xy list site does not have the correct num of tuples'


def test_fit_predict_tysi():
    X_train_test = X2_test.iloc[range(30)]
    y_train_test = y2_test.iloc[range(30)]
    clf_test = RandomForestRegressor(random_state=130)
    val = (X2_test.loc[[5, 9, 15, 19, 25, 29]], X2_test.loc[[35, 39]],
           y2_test.loc[[5, 9, 15, 19, 25, 29]], y2_test.loc[[35, 39]])

    trainpred_test, testpred_test = \
        RFR_LASSO.fit_predict_tysi(clf_test, val, X_train_test, y_train_test)

    assert isinstance(trainpred_test[2], np.float), \
        'predicted value of wrong type, expected: float'
    assert len(testpred_test) == len(val[1]), \
        'num of predicted values != num of input values, they must match'


def test_rmse_tysi():
    val = (X2_test.loc[[5, 9, 15, 19, 25, 29]], X2_test.loc[[35, 39]],
           y2_test.loc[[5, 9, 15, 19, 25, 29]], y2_test.loc[[35, 39]])
    trainpred_test = np.array([7.00974634, 8.03254739, 7.00974634,
                              8.03254739, 7.00974634, 8.03254739])
    testpred_test = np.array([7.00974634, 8.03254739])

    train_rmse_test, test_rmse_test = \
        RFR_LASSO.rmse_tysi(val, trainpred_test, testpred_test)

    assert isinstance(train_rmse_test, np.float), \
        'predicted rmse of wrong type, expected: float'
    assert test_rmse_test > 0, \
        'RMSE cannot be less than 0'


def test_make_dict_tysi():
    val = (X2_test.loc[[5, 9, 15, 19, 25, 29]], X2_test.loc[[35, 39]],
           y2_test.loc[[5, 9, 15, 19, 25, 29]], y2_test.loc[[35, 39]])

    val2 = (X2_test.loc[[1, 2, 11, 12, 21, 22]], X2_test.loc[[31, 32]],
            y2_test.loc[[1, 2, 11, 12, 21, 22]], y2_test.loc[[31, 32]])

    train_rmse_test = 0.05715435
    test_rmse_test = 0.05715435

    train_dict_test = defaultdict(list)
    test_dict_test = defaultdict(list)

    train_dict_test, test_dict_test = \
        RFR_LASSO.make_dict_tysi(val, df2_test, train_rmse_test,
                                 test_rmse_test, train_dict_test,
                                 test_dict_test, 'type')

    train_dict_test2, test_dict_test2 = \
        RFR_LASSO.make_dict_tysi(val2, df2_test, train_rmse_test,
                                 test_rmse_test, train_dict_test,
                                 test_dict_test, 'site')

    assert train_dict_test['train rmse III-V'][0] == train_rmse_test, \
        'train rmse value did not append correctly to dict'
    assert len(test_dict_test) > 0, \
        'test rmse value is empty, should have 1 value'
    assert train_dict_test2['train rmse sub'][0] == train_rmse_test, \
        'train rmse value did not append correctly to dict'
    assert len(train_dict_test2) == len(test_dict_test2), \
        'train dict and test dict are not the same length.'


def test_wrapper_tysi():
    X_train_test = X2_test.iloc[range(30)]
    y_train_test = y2_test.iloc[range(30)]
    clf_test = RandomForestRegressor(random_state=130)

    train_dict_test = defaultdict(list)
    test_dict_test = defaultdict(list)

    train_dict_test2 = defaultdict(list)
    test_dict_test2 = defaultdict(list)

    train_idx_test = np.array(range(30))
    test_idx_test = np.array(range(30, 40))

    train_dict_test, test_dict_test = \
        RFR_LASSO.wrapper_tysi(df2_test, clf_test, X_train_test, y_train_test,
                               train_dict_test, test_dict_test,
                               train_idx_test, test_idx_test, X2_test,
                               y2_test, 'type')

    train_dict_test2, test_dict_test2 = \
        RFR_LASSO.wrapper_tysi(df2_test, clf_test, X_train_test, y_train_test,
                               train_dict_test2, test_dict_test2,
                               train_idx_test, test_idx_test, X2_test,
                               y2_test, 'site')

    assert len(train_dict_test) == 3, \
        'type train dictionary does not have the correct number of entries, \
        got {}, but should be 3'.format(len(train_dict_test))
    assert test_dict_test['test rmse IV-IV'][0] == 0.019287611100909687, \
        'RMSE value inputted into the incorrect sc type'
    assert len(test_dict_test2) == 2, \
        'site test dictionary does not have the correct number of entries, \
        got {}, but should be 2'.format(len(test_dict_test2))
    assert len(train_dict_test2['train rmse int']) == 1, \
        'incorrect number of values for RMSE'


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

    df_26, _, _ = RFR_LASSO.rmse_table_tysi(train_dict_ty, test_dict_ty,
                                            'type')
    _, df_int = RFR_LASSO.rmse_table_tysi(train_dict_si, test_dict_si, 'site')

    assert df_26.columns[0] == 'train rmse II-VI', \
        'incorrect column in type II-VI dataframe, should be train rmse II-VI'
    assert df_26.shape[0] == 2, \
        'length of df is incorrect. Should be 2, instead got {}'.\
        format(df_26.shape[0])
    assert df_int.columns[1] == 'test rmse int', \
        'incorrect column in site int dataframe, should be train rmse int'
    assert isinstance(df_int.iloc[1][0], str), \
        'last row of df should be of type str'


def test_RFR_LASSO_rmse():
    rd = RFR_LASSO.RFR_LASSO_rmse(df2_test, d_start=5, folds=2,
                                  output_type='none')

    assert len(rd) == 3,\
        'rmse table is incorrect length. expected 3, got {}'.\
        format(len(rd))
    assert isinstance(rd.loc[len(rd)-1][0], str), \
        'mean/ stddev row in rmse_table is of wrong type, expected str'


def test_dict_sorter():
    dic_test = {'a': [1, 2, 3], 'b': [1, 2, 3]}

    dft_test, mean_test, std_test = RFR_LASSO.dict_sorter(dic_test)

    assert isinstance(dft_test[0], str), \
        'value in dft list is of the wrong type, should be str.'
    assert len(mean_test) == 2, \
        'incorrect num of values appended to list, expected 2, got {}'.\
        format(len(mean_test))
    assert std_test[1] == 1, \
        'standard deviation calculated incorrectly. expected 1, got {}'.\
        format(std_test[1])


def test_plotdict_tysi():
    test_dict_test = \
        defaultdict(list,
                    {6.979431759: [3.98069889, 0.38048415, 0.28289173],
                     7.095054806: [-0.78001682,  0.20058657,  0.65550337],
                     5.935974344: [-0.9246399, -0.00948319, -0.41550516]})

    t26, _, _ = RFR_LASSO.plotdict_tysi(test_dict_test, df2_test, 'output',
                                        'type')
    tsub, _ = RFR_LASSO.plotdict_tysi(test_dict_test, df2_test, 'output',
                                      'site')

    assert len(t26) == 2, \
        'dictionary is incorrect length. expected 2, got {}'.format(len(t26))
    assert isinstance((list(t26.keys()))[0], np.float), \
        'values are not of correct type, expected np.float64'
    assert list(tsub.keys())[0] < list(tsub.keys())[1], \
        'values not added into dictionary correctly'


def test_zip_to_ddict():
    Y_train_test = [6.97943176, 7.09505481, 5.93597434, 6.97943176,
                    7.09505481, 5.93597434]
    Y_test_test = [6.97943176, 7.09505481, 5.93597434, 6.97943176,
                   7.09505481, 5.93597434]
    PRED_train_test = [3.98069889, 0.38048415, 0.28289173, 0.65550337,
                       0.38048415, 0.28289173]
    PRED_test_test = [0.38048415, 0.07535999, 0.07535999, 0.20058657,
                      0.38048415, 0.07535999]

    traind, testd = RFR_LASSO.zip_to_ddict(Y_train_test, Y_test_test,
                                           PRED_train_test, PRED_test_test)

    assert len(traind) == len(testd), \
        'train and test dictionaries are of incorrect length, they\
         should be the same length'
    assert list(testd.keys())[0] > 6.0, \
        'first key in test df is of incorrect value, should be > 6.0'


# default dictionaries to use for test_plot_sorter_none/type/site
traind = \
    defaultdict(list,
                {6.979431759: [3.98069889, 0.38048415, 0.28289173],
                 7.095054806: [-0.78001682,  0.20058657,  0.65550337],
                 5.935974344: [-0.9246399, -0.00948319, -0.41550516]})
testd = \
    defaultdict(list,
                {6.979431759: [3.98069889, 0.38048415, 0.28289173],
                 7.095054806: [-0.78001682,  0.20058657,  0.65550337],
                 5.935974344: [-0.9246399, -0.00948319, -0.41550516]})


def test_plot_sorter_none():

    traindf, _ = RFR_LASSO.plot_sorter_none(traind, testd, 'output')

    assert traindf.columns[0] == 'dft_train', \
        "train df first column is incorrect, should be 'dft_train'"
    assert traindf.loc[1][0] == list(traind.keys())[1],\
        'dataframe values do not match dictionary'


def test_plot_sorter_type():

    tdf, _, _, df44 = RFR_LASSO.plot_sorter_type(traind, testd, 'output',
                                                 df2_test, 'type')

    assert len(df44) == 1, \
        'incorrect length of df. expected 1, got {}'.format(len(df44))
    assert len(tdf) > len(df44), \
        'train df should be longer than individual test dfs'


def test_plot_sorter_site():

    tdf, dfsub, _ = RFR_LASSO.plot_sorter_site(traind, testd, 'output',
                                               df2_test, 'site')

    assert len(dfsub) == 2, \
        'incorrect length of df. expected 2, got {}'.format(len(dfsub))
    assert len(tdf) > len(dfsub), \
        'train df should be longer than individual test dfs'
    assert dfsub.columns[2] == 'stddev_test_sub', \
        'test sub df column 3 has the wrong name. expected stddev_test_sub,\
         got {}'.format(dfsub.columns[2])


def test_RFR_LASSO_plot():
    train_table, test_table, rmse_table = \
        RFR_LASSO.RFR_LASSO_plot(df_test, d_start=5, folds=2,
                                 output_type='none')

    assert len(train_table) == len(test_table),\
        'train and test table should be the same length'
    assert isinstance(rmse_table.loc[len(rmse_table)-1][0], str), \
        'mean/ stddev row in rmse_table is of wrong type, expected str'
