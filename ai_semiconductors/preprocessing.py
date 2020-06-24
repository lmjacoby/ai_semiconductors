#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# import sys
# sys.path.append("./data/")


def preprocessing(Impurity, AB, Site):
    '''
    extract information from dataframe based on
    semiconductor impurity, compound, defect site
    '''
    import numpy as np
    import pandas as pd
    # read prediction data and DFT data
    KRR = pd.read_csv('./data/Prediction_KRR.csv')
    RFR = pd.read_csv('./data/Prediction_RFR.csv')
    NN = pd.read_csv('./data/Prediction_NN.csv')
    Fullchem = pd.read_csv('./data/ML_data_v2.csv')
    DFT = pd.read_csv('./data/DFT.csv')
    values = ['dH (A-rich)', 'dH (B-rich)', '(+3/+2)', '(+2/+1)',
              '(+1/0)', '(0/-1)', '(-1/-2)', '(-2/-3)']
    std = ['dH (A-rich) std', 'dH (B-rich) std', '(+3/+2) std',
           '(+2/+1) std', '(+1/0) std', '(0/-1) std', '(-1/-2) std',
           '(-2/-3) std']
    try:
        KRR_output = np.array(KRR[(KRR['M'] == Impurity) &
                                  (KRR['AB'] == AB) &
                                  (KRR['Site'] == Site)][values])[0]
        KRR_std = np.array(KRR[(KRR['M'] == Impurity) &
                           (KRR['AB'] == AB) &
                           (KRR['Site'] == Site)][std])[0]
    except (IndexError, ValueError, TypeError):
        print("We don't calculate this type of semiconductor/Impurity yet")
        raise ValueError('Try to input again')
        return
    RFR_output = np.array(RFR[(RFR['Impurity'] == Impurity) &
                              (RFR['AB'] == AB) &
                              (RFR['Site'] == Site)][values])[0]
    RFR_std = np.array(RFR[(RFR['Impurity'] == Impurity) &
                           (RFR['AB'] == AB) &
                           (RFR['Site'] == Site)][std])[0]
    NN_output = np.array(NN[(NN['M'] == Impurity) &
                            (NN['AB'] == AB) &
                            (NN['Site'] == Site)][values])[0]
    NN_std = np.array(NN[(NN['M'] == Impurity) &
                         (NN['AB'] == AB) &
                         (NN['Site'] == Site)][std])[0]
    try:
        DFT_values = np.array(DFT[(DFT['Impurity'] == Impurity) &
                                  (DFT['AB'] == AB) &
                                  (DFT['Site'] == Site)][values])[0]
        DFT_std = np.zeros(len(DFT_values))
        DFT_exist = True
    except IndexError:
        print("We don't have DFT values for",
              "this type of semiconductor/Impurity yet")
        DFT_exist = False
    PBE_gap = Fullchem[(Fullchem['M'] == Impurity) &
                       (Fullchem['AB'] == AB) &
                       (Fullchem['Site'] == Site)]['PBE_gap'].tolist()[0]
    TL_energies = []
    TL_std = []
    for i in range(2, len(KRR_output)):
        if DFT_exist:
            TL_energies.append(DFT_values[i])
            TL_std.append(DFT_std[i])
        TL_energies.append(KRR_output[i])
        TL_energies.append(RFR_output[i])
        TL_energies.append(NN_output[i])

        TL_std.append(KRR_std[i])
        TL_std.append(RFR_std[i])
        TL_std.append(NN_std[i])
    FM_energies = []
    FM_std = []
    for i in range(0, 2):
        if DFT_exist:
            FM_energies.append(DFT_values[i])
            FM_std.append(DFT_std[i])
        FM_energies.append(KRR_output[i])
        FM_energies.append(RFR_output[i])
        FM_energies.append(NN_output[i])
        FM_std.append(KRR_std[i])
        FM_std.append(RFR_std[i])
        FM_std.append(NN_std[i])
    return TL_energies, TL_std, FM_energies, FM_std, PBE_gap, DFT_exist
