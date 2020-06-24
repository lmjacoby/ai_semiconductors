import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise, GaussianDropout
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from math import sqrt
from matplotlib.lines import Line2D
from collections import defaultdict


class NeuralNetwork:
    def data_prep(df, predict=True):
        """
        Function adds two additional descriptor columns for sc type and defect
        site, then standardized and one-hot encodes input and output
        """
        # load labels df - sc type, defect site, compound
        labels = pd.read_csv("labels.csv", sep="|")
        labels.drop(columns=labels.iloc[:, 1:2], inplace=True)

        # encode sc type and defect site and add to df
        enc = OrdinalEncoder(dtype=np.int)
        enc_labels = enc.fit_transform(labels)
        labels = pd.DataFrame(enc_labels, columns=["Type", "Site"])
        df = pd.concat([df, labels], axis=1)

        # separate categorical and continuous data
        categorical = pd.DataFrame()
        continuous = pd.DataFrame()

        # seperate x and y values
        y_fake = df.iloc[:, 0:2]  # HA and HB unused additional columns
        y = df.iloc[:, 2:8]
        x = df.iloc[:, 8:1000000]

        for column in x.columns:
            if(x[column].dtypes == "int"):
                categorical[column] = x[column]
            elif(x[column].dtypes == "float"):
                continuous[column] = x[column]
            else:
                pass

        # one hot encode categorical data
        onehotencoder = OneHotEncoder()
        categorical = onehotencoder.fit_transform(categorical).toarray()

        # standardize continuous data
        input_scaler = StandardScaler()
        continuous = input_scaler.fit_transform(continuous)

        # re-combine categorical and continuous x values
        x = np.concatenate((continuous, categorical), axis=1)

        # standardize outputs (DFT predicted / output)
        output_scaler = StandardScaler()
        y = output_scaler.fit_transform(y)

        # concatenate x and y back to df
        df = pd.DataFrame(np.concatenate((y_fake, y, x), axis=1))

        return (df, output_scaler, input_scaler)

    def graph_results(epochs, loss, val_loss, dft_train_P32, mean_train_P32,
                      std_train_P32, dft_test_P32, mean_test_P32, std_test_P32,
                      dft_train_P21, mean_train_P21, std_train_P21,
                      dft_test_P21, mean_test_P21, std_test_P21, dft_train_P10,
                      mean_train_P10, std_train_P10, dft_test_P10,
                      mean_test_P10, std_test_P10, dft_train_N01,
                      mean_train_N01, std_train_N01, dft_test_N01,
                      mean_test_N01, std_test_N01, dft_train_N12,
                      mean_train_N12, std_train_N12, dft_test_N12,
                      mean_test_N12, std_test_N12, dft_train_N23,
                      mean_train_N23, std_train_N23, dft_test_N23,
                      mean_test_N23, std_test_N23):

        test_color = "#ff5042"   # red
        train_color = "#080bb6"  # blue
        error_color = "#bababa"  # grey
        fig, ax = plt.subplots(figsize=(12, 7))
        x_plot = np.arange(0, epochs)
        for index in range(loss.shape[0]):
            ax.plot(x_plot, loss[index], label="Training Loss",
                    color=train_color, lw=2)
            ax.plot(x_plot, val_loss[index], label="Validation Loss",
                    color=test_color, lw=2)
        ax.set_xlabel("Epoch Number", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.set_ylim(0, 3)
        ax.set_title('Training/Validation Loss Functions', fontsize=20)
        legend_lines = [Line2D([0], [0], color=train_color, lw=4),
                        Line2D([0], [0], color=test_color, lw=4)]
        ax.legend(legend_lines, ["Loss", "Val. Loss"], fontsize=20)
        plt.show()

        #############################################################
        # plot results
        fig, ax = plt.subplots(2, 3, figsize=(22, 10))
        fig.subplots_adjust(hspace=.25, wspace=0.175, top=.90)
        fig.suptitle("Transition Levels: NN Predictions VS. DFT Calc.",
                     fontsize=20, va='top')
        fig.text(0.5, 0.05, 'DFT Calculations (eV)', ha='center', fontsize=17)
        fig.text(0.075, 0.5, 'Neural Network Prediction (eV)', va='center',
                 rotation='vertical', fontsize=17)

        subtitle_size = 17
        test_alpha = 0.65
        train_alpha = 0.85
        scatter_size = 6.5

        # P32
        ax[0, 0].set_title("Transition Level: (+3/+2)", fontsize=subtitle_size)
        ax[0, 0].errorbar(dft_train_P32, mean_train_P32, yerr=std_train_P32,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[0, 0].errorbar(dft_test_P32, mean_test_P32, yerr=std_test_P32,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[0, 0].get_xlim(), ax[0, 0].get_ylim()]),
                np.max([ax[0, 0].get_xlim(), ax[0, 0].get_ylim()])]
        ax[0, 0].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[0, 0].set_xlim(lims)
        ax[0, 0].set_ylim(lims)
        ax[0, 0].legend(fontsize=subtitle_size)

        # P21
        ax[1, 0].set_title("Transition Level: (+2/+1)", fontsize=subtitle_size)
        ax[1, 0].errorbar(dft_train_P21, mean_train_P21, yerr=std_train_P21,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[1, 0].errorbar(dft_test_P21, mean_test_P21, yerr=std_test_P21,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[1, 0].get_xlim(), ax[1, 0].get_ylim()]),
                np.max([ax[1, 0].get_xlim(), ax[1, 0].get_ylim()])]
        ax[1, 0].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[1, 0].set_xlim(lims)
        ax[1, 0].set_ylim(lims)

        # P10
        ax[0, 1].set_title("Transition Level: (+1/0)", fontsize=subtitle_size)
        ax[0, 1].errorbar(dft_train_P10, mean_train_P10, yerr=std_train_P10,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[0, 1].errorbar(dft_test_P10, mean_test_P10, yerr=std_test_P10,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[0, 1].get_xlim(), ax[0, 1].get_ylim()]),
                np.max([ax[0, 1].get_xlim(), ax[0, 1].get_ylim()])]
        ax[0, 1].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[0, 1].set_xlim(lims)
        ax[0, 1].set_ylim(lims)

        # N01
        ax[1, 1].set_title("Transition Level: (0/-1)", fontsize=subtitle_size)
        ax[1, 1].errorbar(dft_train_N01, mean_train_N01, yerr=std_train_N01,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[1, 1].errorbar(dft_test_N01, mean_test_N01, yerr=std_test_N01,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()]),
                np.max([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()])]
        ax[1, 1].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[1, 1].set_xlim(lims)
        ax[1, 1].set_ylim(lims)

        # N12
        ax[0, 2].set_title("Transition Level: (-1/-2)", fontsize=subtitle_size)
        ax[0, 2].errorbar(dft_train_N12, mean_train_N12, yerr=std_train_N12,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[0, 2].errorbar(dft_test_N12, mean_test_N12, yerr=std_test_N12,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[0, 2].get_xlim(), ax[0, 2].get_ylim()]),
                np.max([ax[0, 2].get_xlim(), ax[0, 2].get_ylim()])]
        ax[0, 2].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[0, 2].set_xlim(lims)
        ax[0, 2].set_ylim(lims)

        # N23
        ax[1, 2].set_title("Transition Level: (-2/-3)", fontsize=subtitle_size)
        ax[1, 2].errorbar(dft_train_N23, mean_train_N23, yerr=std_train_N23,
                          fmt='o', label="Train", ecolor=error_color,
                          elinewidth=2.5, alpha=train_alpha, color=train_color,
                          markersize=scatter_size, zorder=3)
        ax[1, 2].errorbar(dft_test_N23, mean_test_N23, yerr=std_test_N23,
                          fmt='o', label="Test", ecolor=error_color,
                          elinewidth=2.5, alpha=test_alpha, color=test_color,
                          markersize=scatter_size, zorder=3)
        lims = [np.min([ax[1, 2].get_xlim(), ax[1, 2].get_ylim()]),
                np.max([ax[1, 2].get_xlim(), ax[1, 2].get_ylim()])]
        ax[1, 2].plot(lims, lims, color="black", zorder=3, alpha=0.7)
        ax[1, 2].set_xlim(lims)
        ax[1, 2].set_ylim(lims)

        plt.show()

    def eval_catgr(x_test_P32, x_test_P21, x_test_P10, x_test_N01, x_test_N12,
                   x_test_N23, y_test, output_scaler, model):
        y_test = output_scaler.inverse_transform(y_test)

        pred_test = model.predict([x_test_P32, x_test_P21, x_test_P10,
                                   x_test_N01, x_test_N12, x_test_N23])

        pred_test = output_scaler.inverse_transform(pred_test)

        test_RMSE_P32 = sqrt(mean_squared_error(y_test[:, 0], pred_test[:, 0]))
        test_RMSE_P21 = sqrt(mean_squared_error(y_test[:, 1], pred_test[:, 1]))
        test_RMSE_P10 = sqrt(mean_squared_error(y_test[:, 2], pred_test[:, 2]))
        test_RMSE_N01 = sqrt(mean_squared_error(y_test[:, 3], pred_test[:, 3]))
        test_RMSE_N12 = sqrt(mean_squared_error(y_test[:, 4], pred_test[:, 4]))
        test_RMSE_N23 = sqrt(mean_squared_error(y_test[:, 5], pred_test[:, 5]))

        return (test_RMSE_P32, test_RMSE_P21, test_RMSE_P10, test_RMSE_N01,
                test_RMSE_N12, test_RMSE_N23)

    def model_eval(model, prediction, x_train_P32, x_train_P21, x_train_P10,
                   x_train_N01, x_train_N12, x_train_N23, x_test_P32,
                   x_test_P21, x_test_P10, x_test_N01, x_test_N12, x_test_N23,
                   y_train, y_test, output_scaler):
        """
        Prints out the RMSE trian and test values
        """
        y_train = output_scaler.inverse_transform(y_train)
        y_test = output_scaler.inverse_transform(y_test)

        pred_train = model.predict([x_train_P32, x_train_P21, x_train_P10,
                                    x_train_N01, x_train_N12, x_train_N23])
        pred_test = model.predict([x_test_P32, x_test_P21, x_test_P10,
                                   x_test_N01, x_test_N12, x_test_N23])

        pred_train = output_scaler.inverse_transform(pred_train)
        pred_test = output_scaler.inverse_transform(pred_test)

        train_RMSE_P32 = sqrt(mean_squared_error(y_train[:, 0],
                              pred_train[:, 0]))
        train_RMSE_P21 = sqrt(mean_squared_error(y_train[:, 1],
                              pred_train[:, 1]))
        train_RMSE_P10 = sqrt(mean_squared_error(y_train[:, 2],
                              pred_train[:, 2]))
        train_RMSE_N01 = sqrt(mean_squared_error(y_train[:, 3],
                              pred_train[:, 3]))
        train_RMSE_N12 = sqrt(mean_squared_error(y_train[:, 4],
                              pred_train[:, 4]))
        train_RMSE_N23 = sqrt(mean_squared_error(y_train[:, 5],
                              pred_train[:, 5]))
        test_RMSE_P32 = sqrt(mean_squared_error(y_test[:, 0], pred_test[:, 0]))
        test_RMSE_P21 = sqrt(mean_squared_error(y_test[:, 1], pred_test[:, 1]))
        test_RMSE_P10 = sqrt(mean_squared_error(y_test[:, 2], pred_test[:, 2]))
        test_RMSE_N01 = sqrt(mean_squared_error(y_test[:, 3], pred_test[:, 3]))
        test_RMSE_N12 = sqrt(mean_squared_error(y_test[:, 4], pred_test[:, 4]))
        test_RMSE_N23 = sqrt(mean_squared_error(y_test[:, 5], pred_test[:, 5]))

        print("- - - - - - - - - - - - - - - - - - - -")
        print("RMSE Training / Testing (eV):")
        print("(+3/+2): %.4f / %.4f" % (train_RMSE_P32, test_RMSE_P32))
        print("(+2/+1): %.4f / %.4f" % (train_RMSE_P21, test_RMSE_P21))
        print("(+1/0): %.4f / %.4f" % (train_RMSE_P10, test_RMSE_P10))
        print("(0/-1): %.4f / %.4f" % (train_RMSE_N01, test_RMSE_N01))
        print("(-1/-2): %.4f / %.4f" % (train_RMSE_N12, test_RMSE_N12))
        print("(-2/-3): %.4f / %.4f" % (train_RMSE_N23, test_RMSE_N23))
        print("- - - - - - - - - - - - - - - - - - - -")
        low_epoch = (np.argmin(prediction.history["val_loss"]) + 1)
        low_val_loss = np.amin(prediction.history["val_loss"])
        low_epoch_train = (np.argmin(prediction.history["loss"]) + 1)
        low_val_loss_train = np.amin(prediction.history["loss"])
        print("Lowest Val. loss: %.4f at %s epochs" % (low_val_loss,
                                                       low_epoch))
        print("Lowest train loss: %.4f at %s epochs" % (low_val_loss_train,
                                                        low_epoch_train))
        print("- - - - - - - - - - - - - - - - - - - -")
        print("")

        return (train_RMSE_P32, train_RMSE_P21, train_RMSE_P10, train_RMSE_N01,
                train_RMSE_N12, train_RMSE_N23, test_RMSE_P32, test_RMSE_P21,
                test_RMSE_P10, test_RMSE_N01, test_RMSE_N12, test_RMSE_N23,
                pred_train, pred_test, y_train, y_test)

    def graph_prep(Y_train, Y_test, PRED_train, PRED_test):
        # Combine training and testing datasets into dictionary
        Y_train = list(Y_train)
        Y_test = list(Y_test)
        PRED_train = list(PRED_train)
        PRED_test = list(PRED_test)

        train_zip = list(zip(Y_train, PRED_train))
        test_zip = list(zip(Y_test, PRED_test))
        train_dic = defaultdict(list)
        test_dic = defaultdict(list)

        for y_training, pred_training in train_zip:
            train_dic[y_training].append(pred_training)

        for y_testing, pred_testing in test_zip:
            test_dic[y_testing].append(pred_testing)

        dft_train = np.empty(0)
        mean_train = np.empty(0)
        std_train = np.empty(0)
        dft_test = np.empty(0)
        mean_test = np.empty(0)
        std_test = np.empty(0)

        # calculate and append meand and stdev for each dft datapoint
        for key, values in train_dic.items():
            dft_train = np.append(dft_train, key)
            mean_train = np.append(mean_train, stats.mean(values))
            std_train = np.append(std_train, stats.stdev(values))

        for key, values in test_dic.items():
            dft_test = np.append(dft_test, key)
            mean_test = np.append(mean_test, stats.mean(values))
            std_test = np.append(std_test, stats.stdev(values))

        return (dft_train, dft_test, mean_train, mean_test, std_train,
                std_test)

    def pred_fullchem(df_full, model, input_scaler):
        # load full chem labels df - sc type, defect site, compound
        labels = pd.read_csv("labels_fullchem.csv", sep="|")
        labels.drop(columns=labels.iloc[:, 1:2], inplace=True)

        # encode sc type and defect site and add to df
        enc = OrdinalEncoder(dtype=np.int)
        enc_labels = enc.fit_transform(labels)
        labels = pd.DataFrame(enc_labels, columns=["Type", "Site"])
        x = pd.concat([df_full, labels], axis=1)

        # separate categorical and continuous data
        categorical = pd.DataFrame()
        continuous = pd.DataFrame()

        for column in x.columns:
            if(x[column].dtypes == "int"):
                categorical[column] = x[column]
            elif(x[column].dtypes == "float"):
                continuous[column] = x[column]
            else:
                pass

        # one hot encode categorical data
        onehotencoder = OneHotEncoder()
        categorical = onehotencoder.fit_transform(categorical).toarray()

        # standardize continuous data
        continuous = input_scaler.fit_transform(continuous)

        # re-combine categorical and continuous x values
        x = np.concatenate((continuous, categorical), axis=1)
        x = pd.DataFrame(x)

        x_HA = x.iloc[:, 0:111]
        x_HB = x.iloc[:, 111:258]
        x_P32 = x.iloc[:, 258:346]
        x_P21 = x.iloc[:, 346:424]
        x_P10 = x.iloc[:, 424:465]
        x_N01 = x.iloc[:, 465:532]
        x_N12 = x.iloc[:, 532:580]
        x_N23 = x.iloc[:, 580:667]
        onehot_label = x.iloc[:, 667:675]

        for dff in ([x_HA, x_HB, x_P32, x_P21, x_P10, x_N01, x_N12, x_N23]):
            dff = pd.concat([dff, onehot_label], axis=1)

        full_predict = model.predict([x_P32, x_P21, x_P10, x_N01, x_N12,
                                      x_N23])

        return (full_predict)

    def run_k_fold(df, epochs, bs, lr, decay, dropout, noise, k_reg,
                   hid_layer_neurons, verbose, folds, repeats, rs, graph,
                   output_scaler, input_scaler, df_full, beta1, beta2,
                   amsgrad):
        """
        This functions performs the k_fold stratify split and runs the neural
        network model for predictions.
        """
        des_labels = pd.read_csv("labels.csv", sep="|")
        des_labels.drop(columns=des_labels.iloc[:, 1:2], inplace=True)
        df = pd.concat([des_labels, df], axis=1)

        enc = OrdinalEncoder(dtype=np.int)
        encode_labels = enc.fit_transform(des_labels)
        labels = pd.DataFrame(encode_labels, columns=["Type", "Site"])
        labels = labels.applymap(str)
        labels = labels[["Type", "Site"]].apply(lambda x: ''.join(x), axis=1)

        # encode the new string col to 0-14 (15 total classes - 3 sctypes x 5
        # defsites)
        combined_labels = np.array(labels).reshape(-1, 1)

        total_folds = 0
        fold_num = 0
        train_rmse_P32 = []
        train_rmse_P21 = []
        train_rmse_P10 = []
        train_rmse_N01 = []
        train_rmse_N12 = []
        train_rmse_N23 = []
        test_rmse_P32 = []
        test_rmse_P21 = []
        test_rmse_P10 = []
        test_rmse_N01 = []
        test_rmse_N12 = []
        test_rmse_N23 = []

        sub_test_rmse_P32 = []
        sub_test_rmse_P21 = []
        sub_test_rmse_P10 = []
        sub_test_rmse_N01 = []
        sub_test_rmse_N12 = []
        sub_test_rmse_N23 = []

        int_test_rmse_P32 = []
        int_test_rmse_P21 = []
        int_test_rmse_P10 = []
        int_test_rmse_N01 = []
        int_test_rmse_N12 = []
        int_test_rmse_N23 = []

        IIVI_test_rmse_P32 = []
        IIVI_test_rmse_P21 = []
        IIVI_test_rmse_P10 = []
        IIVI_test_rmse_N01 = []
        IIVI_test_rmse_N12 = []
        IIVI_test_rmse_N23 = []

        IIIV_test_rmse_P32 = []
        IIIV_test_rmse_P21 = []
        IIIV_test_rmse_P10 = []
        IIIV_test_rmse_N01 = []
        IIIV_test_rmse_N12 = []
        IIIV_test_rmse_N23 = []

        IVIV_test_rmse_P32 = []
        IVIV_test_rmse_P21 = []
        IVIV_test_rmse_P10 = []
        IVIV_test_rmse_N01 = []
        IVIV_test_rmse_N12 = []
        IVIV_test_rmse_N23 = []

        loss = []
        val_loss = []

        Y_train_P32 = np.empty(0)
        Y_test_P32 = np.empty(0)
        PRED_train_P32 = np.empty(0)
        PRED_test_P32 = np.empty(0)
        Y_train_P21 = np.empty(0)
        Y_test_P21 = np.empty(0)
        PRED_train_P21 = np.empty(0)
        PRED_test_P21 = np.empty(0)
        Y_train_P10 = np.empty(0)
        Y_test_P10 = np.empty(0)
        PRED_train_P10 = np.empty(0)
        PRED_test_P10 = np.empty(0)
        Y_train_N01 = np.empty(0)
        Y_test_N01 = np.empty(0)
        PRED_train_N01 = np.empty(0)
        PRED_test_N01 = np.empty(0)
        Y_train_N12 = np.empty(0)
        Y_test_N12 = np.empty(0)
        PRED_train_N12 = np.empty(0)
        PRED_test_N12 = np.empty(0)
        Y_train_N23 = np.empty(0)
        Y_test_N23 = np.empty(0)
        PRED_train_N23 = np.empty(0)
        PRED_test_N23 = np.empty(0)

        full_pred_P32 = np.empty(0)
        full_pred_P21 = np.empty(0)
        full_pred_P10 = np.empty(0)
        full_pred_N01 = np.empty(0)
        full_pred_N12 = np.empty(0)
        full_pred_N23 = np.empty(0)

        for random in range(1, repeats+1):
            fold_num += 1
            stratified = StratifiedKFold(n_splits=folds, shuffle=True,
                                         random_state=(random*10))
            for train_index, test_index in stratified.split(df,
                                                            combined_labels):
                total_folds += 1

                train = df.loc[train_index]
                test = df.loc[test_index]

                # train split
                y_train = train.iloc[:, 4:10]
                x_train_P32 = train.iloc[:, 268:356]
                x_train_P21 = train.iloc[:, 356:434]
                x_train_P10 = train.iloc[:, 434:475]
                x_train_N01 = train.iloc[:, 475:542]
                x_train_N12 = train.iloc[:, 542:590]
                x_train_N23 = train.iloc[:, 590:677]

                # test split
                y_test = test.iloc[:, 4:10]
                x_test_P32 = test.iloc[:, 268:356]
                x_test_P21 = test.iloc[:, 356:434]
                x_test_P10 = test.iloc[:, 434:475]
                x_test_N01 = test.iloc[:, 475:542]
                x_test_N12 = test.iloc[:, 542:590]
                x_test_N23 = test.iloc[:, 590:677]

                # sc type and defect site one-hot labels
                onehot_label_train = train.iloc[:, 677:685]
                onehot_label_test = test.iloc[:, 677:685]

                # concat one hot labels with each respective df
                for dff in ([x_train_P32, x_train_P21, x_train_P10,
                             x_train_N01, x_train_N12, x_train_N23]):
                    dff = pd.concat([dff, onehot_label_train], axis=1)

                for dff in ([x_test_P32, x_test_P21, x_test_P10, x_test_N01,
                            x_test_N12, x_test_N23]):
                    dff = pd.concat([dff, onehot_label_test], axis=1)

                # sc type and defect site split
                sub_test_index = ((test.Site == "M_A") +
                                  (test.Site == "M_B"))
                int_test_index = ((test.Site == "M_i_A") +
                                  (test.Site == "M_i_B") +
                                  (test.Site == "M_i_neut"))

                sub_test = test.loc[sub_test_index]
                int_test = test.loc[int_test_index]
                IIVI_test = test.loc[test.Type == "II-VI"]
                IIIV_test = test.loc[test.Type == "III-V"]
                IVIV_test = test.loc[test.Type == "IV-IV"]

                y_sub_test = sub_test.iloc[:, 4:10]
                x_sub_test_P32 = sub_test.iloc[:, 268:356]
                x_sub_test_P21 = sub_test.iloc[:, 356:434]
                x_sub_test_P10 = sub_test.iloc[:, 434:475]
                x_sub_test_N01 = sub_test.iloc[:, 475:542]
                x_sub_test_N12 = sub_test.iloc[:, 542:590]
                x_sub_test_N23 = sub_test.iloc[:, 590:677]
                one_hot_label = sub_test.iloc[:, 677:685]
                for dff in ([x_sub_test_P32, x_sub_test_P21, x_sub_test_P10,
                             x_sub_test_N01, x_sub_test_N12, x_sub_test_N23]):
                    dff = pd.concat([dff, one_hot_label], axis=1)

                y_int_test = int_test.iloc[:, 4:10]
                x_int_test_P32 = int_test.iloc[:, 268:356]
                x_int_test_P21 = int_test.iloc[:, 356:434]
                x_int_test_P10 = int_test.iloc[:, 434:475]
                x_int_test_N01 = int_test.iloc[:, 475:542]
                x_int_test_N12 = int_test.iloc[:, 542:590]
                x_int_test_N23 = int_test.iloc[:, 590:677]
                one_hot_label = int_test.iloc[:, 677:685]
                for dff in ([x_int_test_P32, x_int_test_P21, x_int_test_P10,
                             x_int_test_N01, x_int_test_N12, x_int_test_N23]):
                    dff = pd.concat([dff, one_hot_label], axis=1)

                y_IIVI_test = IIVI_test.iloc[:, 4:10]
                x_IIVI_test_P32 = IIVI_test.iloc[:, 268:356]
                x_IIVI_test_P21 = IIVI_test.iloc[:, 356:434]
                x_IIVI_test_P10 = IIVI_test.iloc[:, 434:475]
                x_IIVI_test_N01 = IIVI_test.iloc[:, 475:542]
                x_IIVI_test_N12 = IIVI_test.iloc[:, 542:590]
                x_IIVI_test_N23 = IIVI_test.iloc[:, 590:677]
                one_hot_label = IIVI_test.iloc[:, 677:685]
                for dff in ([x_IIVI_test_P32, x_IIVI_test_P21, x_IIVI_test_P10,
                             x_IIVI_test_N01, x_IIVI_test_N12,
                             x_IIVI_test_N23]):
                    dff = pd.concat([dff, one_hot_label], axis=1)

                y_IIIV_test = IIIV_test.iloc[:, 4:10]
                x_IIIV_test_P32 = IIIV_test.iloc[:, 268:356]
                x_IIIV_test_P21 = IIIV_test.iloc[:, 356:434]
                x_IIIV_test_P10 = IIIV_test.iloc[:, 434:475]
                x_IIIV_test_N01 = IIIV_test.iloc[:, 475:542]
                x_IIIV_test_N12 = IIIV_test.iloc[:, 542:590]
                x_IIIV_test_N23 = IIIV_test.iloc[:, 590:677]
                one_hot_label = IIIV_test.iloc[:, 677:685]
                for dff in ([x_IIIV_test_P32, x_IIIV_test_P21, x_IIIV_test_P10,
                             x_IIIV_test_N01, x_IIIV_test_N12,
                             x_IIIV_test_N23]):
                    dff = pd.concat([dff, one_hot_label], axis=1)

                y_IVIV_test = IVIV_test.iloc[:, 4:10]
                x_IVIV_test_P32 = IVIV_test.iloc[:, 268:356]
                x_IVIV_test_P21 = IVIV_test.iloc[:, 356:434]
                x_IVIV_test_P10 = IVIV_test.iloc[:, 434:475]
                x_IVIV_test_N01 = IVIV_test.iloc[:, 475:542]
                x_IVIV_test_N12 = IVIV_test.iloc[:, 542:590]
                x_IVIV_test_N23 = IVIV_test.iloc[:, 590:677]
                one_hot_label = IVIV_test.iloc[:, 677:685]
                for dff in ([x_IVIV_test_P32, x_IVIV_test_P21, x_IVIV_test_P10,
                             x_IVIV_test_N01, x_IVIV_test_N12,
                             x_IVIV_test_N23]):
                    dff = pd.concat([dff, one_hot_label], axis=1)

                in_dim_P32 = x_train_P32.shape[1]
                in_dim_P21 = x_train_P21.shape[1]
                in_dim_P10 = x_train_P10.shape[1]
                in_dim_N01 = x_train_N01.shape[1]
                in_dim_N12 = x_train_N12.shape[1]
                in_dim_N23 = x_train_N23.shape[1]

                (model, prediction
                 ) = NeuralNetwork.train_model(x_train_P32, x_train_P21,
                                               x_train_P10, x_train_N01,
                                               x_train_N12, x_train_N23,
                                               y_train,
                                               x_test_P32, x_test_P21,
                                               x_test_P10, x_test_N01,
                                               x_test_N12, x_test_N23,
                                               y_test,
                                               in_dim_P32, in_dim_P21,
                                               in_dim_P10, in_dim_N01,
                                               in_dim_N12, in_dim_N23, epochs,
                                               bs, lr, decay, dropout, noise,
                                               k_reg, hid_layer_neurons,
                                               verbose, beta1, beta2, amsgrad)

                # print RMSE values
                print("K-Fold repeat #: " + str(fold_num))
                print("K-Fold ovrall #: " + str(total_folds))
                (train_RMSE_P32, train_RMSE_P21, train_RMSE_P10,
                 train_RMSE_N01, train_RMSE_N12, train_RMSE_N23, test_RMSE_P32,
                 test_RMSE_P21, test_RMSE_P10, test_RMSE_N01, test_RMSE_N12,
                 test_RMSE_N23, pred_train, pred_test, y_train,
                 y_test) = NeuralNetwork.model_eval(model, prediction,
                                                    x_train_P32, x_train_P21,
                                                    x_train_P10, x_train_N01,
                                                    x_train_N12, x_train_N23,
                                                    x_test_P32, x_test_P21,
                                                    x_test_P10, x_test_N01,
                                                    x_test_N12, x_test_N23,
                                                    y_train, y_test,
                                                    output_scaler)
                # substitutional site evaluation
                (test_sub_RMSE_P32, test_sub_RMSE_P21, test_sub_RMSE_P10,
                 test_sub_RMSE_N01, test_sub_RMSE_N12,
                 test_sub_RMSE_N23) = NeuralNetwork.eval_catgr(x_sub_test_P32,
                                                               x_sub_test_P21,
                                                               x_sub_test_P10,
                                                               x_sub_test_N01,
                                                               x_sub_test_N12,
                                                               x_sub_test_N23,
                                                               y_sub_test,
                                                               output_scaler,
                                                               model)
                # interstitial site evaluation
                (test_int_RMSE_P32, test_int_RMSE_P21, test_int_RMSE_P10,
                 test_int_RMSE_N01, test_int_RMSE_N12,
                 test_int_RMSE_N23) = NeuralNetwork.eval_catgr(x_int_test_P32,
                                                               x_int_test_P21,
                                                               x_int_test_P10,
                                                               x_int_test_N01,
                                                               x_int_test_N12,
                                                               x_int_test_N23,
                                                               y_int_test,
                                                               output_scaler,
                                                               model)

                # IIVI type evaluation
                (test_IIVI_RMSE_P32, test_IIVI_RMSE_P21, test_IIVI_RMSE_P10,
                 test_IIVI_RMSE_N01, test_IIVI_RMSE_N12, test_IIVI_RMSE_N23
                 ) = NeuralNetwork.eval_catgr(x_IIVI_test_P32, x_IIVI_test_P21,
                                              x_IIVI_test_P10, x_IIVI_test_N01,
                                              x_IIVI_test_N12, x_IIVI_test_N23,
                                              y_IIVI_test, output_scaler,
                                              model)
                # IIIV type evaluation
                (test_IIIV_RMSE_P32, test_IIIV_RMSE_P21, test_IIIV_RMSE_P10,
                 test_IIIV_RMSE_N01, test_IIIV_RMSE_N12, test_IIIV_RMSE_N23
                 ) = NeuralNetwork.eval_catgr(x_IIIV_test_P32, x_IIIV_test_P21,
                                              x_IIIV_test_P10, x_IIIV_test_N01,
                                              x_IIIV_test_N12, x_IIIV_test_N23,
                                              y_IIIV_test, output_scaler,
                                              model)
                # IVIV type evaluation
                (test_IVIV_RMSE_P32, test_IVIV_RMSE_P21, test_IVIV_RMSE_P10,
                 test_IVIV_RMSE_N01, test_IVIV_RMSE_N12, test_IVIV_RMSE_N23
                 ) = NeuralNetwork.eval_catgr(x_IVIV_test_P32, x_IVIV_test_P21,
                                              x_IVIV_test_P10, x_IVIV_test_N01,
                                              x_IVIV_test_N12, x_IVIV_test_N23,
                                              y_IVIV_test, output_scaler,
                                              model)

                # Predict full 12k points
                full_predict = NeuralNetwork.pred_fullchem(df_full, model,
                                                           input_scaler)

                full_predict = output_scaler.inverse_transform(full_predict)
                full_predict = np.array(full_predict)
                full_predict_P32 = full_predict[:, 0]
                full_predict_P21 = full_predict[:, 1]
                full_predict_P10 = full_predict[:, 2]
                full_predict_N01 = full_predict[:, 3]
                full_predict_N12 = full_predict[:, 4]
                full_predict_N23 = full_predict[:, 5]
                full_pred_P32 = np.append(full_pred_P32, full_predict_P32)
                full_pred_P21 = np.append(full_pred_P21, full_predict_P21)
                full_pred_P10 = np.append(full_pred_P10, full_predict_P10)
                full_pred_N01 = np.append(full_pred_N01, full_predict_N01)
                full_pred_N12 = np.append(full_pred_N12, full_predict_N12)
                full_pred_N23 = np.append(full_pred_N23, full_predict_N23)

                # append each train and test RMSE
                train_rmse_P32.append(train_RMSE_P32)
                train_rmse_P21.append(train_RMSE_P21)
                train_rmse_P10.append(train_RMSE_P10)
                train_rmse_N01.append(train_RMSE_N01)
                train_rmse_N12.append(train_RMSE_N12)
                train_rmse_N23.append(train_RMSE_N23)
                test_rmse_P32.append(test_RMSE_P32)
                test_rmse_P21.append(test_RMSE_P21)
                test_rmse_P10.append(test_RMSE_P10)
                test_rmse_N01.append(test_RMSE_N01)
                test_rmse_N12.append(test_RMSE_N12)
                test_rmse_N23.append(test_RMSE_N23)

                sub_test_rmse_P32.append(test_sub_RMSE_P32)
                sub_test_rmse_P21.append(test_sub_RMSE_P21)
                sub_test_rmse_P10.append(test_sub_RMSE_P10)
                sub_test_rmse_N01.append(test_sub_RMSE_N01)
                sub_test_rmse_N12.append(test_sub_RMSE_N12)
                sub_test_rmse_N23.append(test_sub_RMSE_N23)

                int_test_rmse_P32.append(test_int_RMSE_P32)
                int_test_rmse_P21.append(test_int_RMSE_P21)
                int_test_rmse_P10.append(test_int_RMSE_P10)
                int_test_rmse_N01.append(test_int_RMSE_N01)
                int_test_rmse_N12.append(test_int_RMSE_N12)
                int_test_rmse_N23.append(test_int_RMSE_N23)

                IIVI_test_rmse_P32.append(test_IIVI_RMSE_P32)
                IIVI_test_rmse_P21.append(test_IIVI_RMSE_P21)
                IIVI_test_rmse_P10.append(test_IIVI_RMSE_P10)
                IIVI_test_rmse_N01.append(test_IIVI_RMSE_N01)
                IIVI_test_rmse_N12.append(test_IIVI_RMSE_N12)
                IIVI_test_rmse_N23.append(test_IIVI_RMSE_N23)

                IIIV_test_rmse_P32.append(test_IIIV_RMSE_P32)
                IIIV_test_rmse_P21.append(test_IIIV_RMSE_P21)
                IIIV_test_rmse_P10.append(test_IIIV_RMSE_P10)
                IIIV_test_rmse_N01.append(test_IIIV_RMSE_N01)
                IIIV_test_rmse_N12.append(test_IIIV_RMSE_N12)
                IIIV_test_rmse_N23.append(test_IIIV_RMSE_N23)

                IVIV_test_rmse_P32.append(test_IVIV_RMSE_P32)
                IVIV_test_rmse_P21.append(test_IVIV_RMSE_P21)
                IVIV_test_rmse_P10.append(test_IVIV_RMSE_P10)
                IVIV_test_rmse_N01.append(test_IVIV_RMSE_N01)
                IVIV_test_rmse_N12.append(test_IVIV_RMSE_N12)
                IVIV_test_rmse_N23.append(test_IVIV_RMSE_N23)

                # loss functions
                loss.append(prediction.history["loss"])
                val_loss.append(prediction.history["val_loss"])

                # appending train and test results
                y_train = np.array(y_train)
                y_train_P32 = y_train[:, 0]
                y_train_P21 = y_train[:, 1]
                y_train_P10 = y_train[:, 2]
                y_train_N01 = y_train[:, 3]
                y_train_N12 = y_train[:, 4]
                y_train_N23 = y_train[:, 5]

                y_test = np.array(y_test)
                y_test_P32 = y_test[:, 0]
                y_test_P21 = y_test[:, 1]
                y_test_P10 = y_test[:, 2]
                y_test_N01 = y_test[:, 3]
                y_test_N12 = y_test[:, 4]
                y_test_N23 = y_test[:, 5]

                pred_train = np.array(pred_train)
                pred_train_P32 = pred_train[:, 0]
                pred_train_P21 = pred_train[:, 1]
                pred_train_P10 = pred_train[:, 2]
                pred_train_N01 = pred_train[:, 3]
                pred_train_N12 = pred_train[:, 4]
                pred_train_N23 = pred_train[:, 5]

                pred_test = np.array(pred_test)
                pred_test_P32 = pred_test[:, 0]
                pred_test_P21 = pred_test[:, 1]
                pred_test_P10 = pred_test[:, 2]
                pred_test_N01 = pred_test[:, 3]
                pred_test_N12 = pred_test[:, 4]
                pred_test_N23 = pred_test[:, 5]

                Y_train_P32 = np.append(Y_train_P32, y_train_P32)
                Y_train_P21 = np.append(Y_train_P21, y_train_P21)
                Y_train_P10 = np.append(Y_train_P10, y_train_P10)
                Y_train_N01 = np.append(Y_train_N01, y_train_N01)
                Y_train_N12 = np.append(Y_train_N12, y_train_N12)
                Y_train_N23 = np.append(Y_train_N23, y_train_N23)

                Y_test_P32 = np.append(Y_test_P32, y_test_P32)
                Y_test_P21 = np.append(Y_test_P21, y_test_P21)
                Y_test_P10 = np.append(Y_test_P10, y_test_P10)
                Y_test_N01 = np.append(Y_test_N01, y_test_N01)
                Y_test_N12 = np.append(Y_test_N12, y_test_N12)
                Y_test_N23 = np.append(Y_test_N23, y_test_N23)

                PRED_train_P32 = np.append(PRED_train_P32, pred_train_P32)
                PRED_train_P21 = np.append(PRED_train_P21, pred_train_P21)
                PRED_train_P10 = np.append(PRED_train_P10, pred_train_P10)
                PRED_train_N01 = np.append(PRED_train_N01, pred_train_N01)
                PRED_train_N12 = np.append(PRED_train_N12, pred_train_N12)
                PRED_train_N23 = np.append(PRED_train_N23, pred_train_N23)

                PRED_test_P32 = np.append(PRED_test_P32, pred_test_P32)
                PRED_test_P21 = np.append(PRED_test_P21, pred_test_P21)
                PRED_test_P10 = np.append(PRED_test_P10, pred_test_P10)
                PRED_test_N01 = np.append(PRED_test_N01, pred_test_N01)
                PRED_test_N12 = np.append(PRED_test_N12, pred_test_N12)
                PRED_test_N23 = np.append(PRED_test_N23, pred_test_N23)

        # reshape loss functions to have length of # of epochs for plotting
        loss = np.array(loss).reshape(-1, epochs)
        val_loss = np.array(val_loss).reshape(-1, epochs)

        # reshape and calculate uncertainties for full 12k points
        full_pred_P32 = full_pred_P32.reshape(-1, repeats*folds)
        full_pred_P21 = full_pred_P21.reshape(-1, repeats*folds)
        full_pred_P10 = full_pred_P10.reshape(-1, repeats*folds)
        full_pred_N01 = full_pred_N01.reshape(-1, repeats*folds)
        full_pred_N12 = full_pred_N12.reshape(-1, repeats*folds)
        full_pred_N23 = full_pred_N23.reshape(-1, repeats*folds)
        stdev_P32 = pd.DataFrame(full_pred_P32.std(axis=1),
                                 columns=["(+3/+2) std"])
        stdev_P21 = pd.DataFrame(full_pred_P21.std(axis=1),
                                 columns=["(+2/+1) std"])
        stdev_P10 = pd.DataFrame(full_pred_P10.std(axis=1),
                                 columns=["(+1/0) std"])
        stdev_N01 = pd.DataFrame(full_pred_N01.std(axis=1),
                                 columns=["(0/-1) std"])
        stdev_N12 = pd.DataFrame(full_pred_N12.std(axis=1),
                                 columns=["(-1/-2) std"])
        stdev_N23 = pd.DataFrame(full_pred_N23.std(axis=1),
                                 columns=["(-2/-3) std"])

        mean_P32 = pd.DataFrame(full_pred_P32.mean(axis=1),
                                columns=["(+3/+2) mean"])
        mean_P21 = pd.DataFrame(full_pred_P21.mean(axis=1),
                                columns=["(+2/+1) mean"])
        mean_P10 = pd.DataFrame(full_pred_P10.mean(axis=1),
                                columns=["(+1/0) mean"])
        mean_N01 = pd.DataFrame(full_pred_N01.mean(axis=1),
                                columns=["(0/-1) mean"])
        mean_N12 = pd.DataFrame(full_pred_N12.mean(axis=1),
                                columns=["(-1/-2) mean"])
        mean_N23 = pd.DataFrame(full_pred_N23.mean(axis=1),
                                columns=["(-2/-3) mean"])

        ovr_predictions = pd.concat([mean_P32, stdev_P32, mean_P21, stdev_P21,
                                     mean_P10, stdev_P10, mean_N01, stdev_N01,
                                     mean_N12, stdev_N12, mean_N23, stdev_N23
                                     ], axis=1)
        ovr_predictions.to_excel(r"Chem_space_NN_TL2.xlsx", index=False)

        # Combine training and testing datasets into dictionary
        (dft_train_P32, dft_test_P32,
         mean_train_P32, mean_test_P32,
         std_train_P32,
         std_test_P32) = NeuralNetwork.graph_prep(Y_train_P32, Y_test_P32,
                                                  PRED_train_P32,
                                                  PRED_test_P32)
        (dft_train_P21, dft_test_P21,
         mean_train_P21, mean_test_P21,
         std_train_P21,
         std_test_P21) = NeuralNetwork.graph_prep(Y_train_P21, Y_test_P21,
                                                  PRED_train_P21,
                                                  PRED_test_P21)
        (dft_train_P10, dft_test_P10,
         mean_train_P10, mean_test_P10,
         std_train_P10,
         std_test_P10) = NeuralNetwork.graph_prep(Y_train_P10, Y_test_P10,
                                                  PRED_train_P10,
                                                  PRED_test_P10)
        (dft_train_N01, dft_test_N01,
         mean_train_N01, mean_test_N01,
         std_train_N01,
         std_test_N01) = NeuralNetwork.graph_prep(Y_train_N01, Y_test_N01,
                                                  PRED_train_N01,
                                                  PRED_test_N01)
        (dft_train_N12, dft_test_N12,
         mean_train_N12, mean_test_N12,
         std_train_N12,
         std_test_N12) = NeuralNetwork.graph_prep(Y_train_N12, Y_test_N12,
                                                  PRED_train_N12,
                                                  PRED_test_N12)
        (dft_train_N23, dft_test_N23,
         mean_train_N23, mean_test_N23,
         std_train_N23,
         std_test_N23) = NeuralNetwork.graph_prep(Y_train_N23, Y_test_N23,
                                                  PRED_train_N23,
                                                  PRED_test_N23)

        # graph loss functions
        if graph is True:
            NeuralNetwork.graph_results(epochs, loss, val_loss,
                                        dft_train_P32, mean_train_P32,
                                        std_train_P32, dft_test_P32,
                                        mean_test_P32, std_test_P32,
                                        dft_train_P21, mean_train_P21,
                                        std_train_P21, dft_test_P21,
                                        mean_test_P21, std_test_P21,
                                        dft_train_P10, mean_train_P10,
                                        std_train_P10, dft_test_P10,
                                        mean_test_P10, std_test_P10,
                                        dft_train_N01, mean_train_N01,
                                        std_train_N01, dft_test_N01,
                                        mean_test_N01, std_test_N01,
                                        dft_train_N12, mean_train_N12,
                                        std_train_N12, dft_test_N12,
                                        mean_test_N12, std_test_N12,
                                        dft_train_N23, mean_train_N23,
                                        std_train_N23, dft_test_N23,
                                        mean_test_N23, std_test_N23)
        else:
            pass

        error_train_P32 = abs(dft_train_P32 - mean_train_P32)
        error_train_P21 = abs(dft_train_P21 - mean_train_P21)
        error_train_P10 = abs(dft_train_P10 - mean_train_P10)
        error_train_N01 = abs(dft_train_N01 - mean_train_N01)
        error_train_N12 = abs(dft_train_N12 - mean_train_N12)
        error_train_N23 = abs(dft_train_N23 - mean_train_N23)

        error_test_P32 = abs(dft_test_P32 - mean_test_P32)
        error_test_P21 = abs(dft_test_P21 - mean_test_P21)
        error_test_P10 = abs(dft_test_P10 - mean_test_P10)
        error_test_N01 = abs(dft_test_N01 - mean_test_N01)
        error_test_N12 = abs(dft_test_N12 - mean_test_N12)
        error_test_N23 = abs(dft_test_N23 - mean_test_N23)

        # calculate mean and stdev of train and test RMSE and display df
        # summary = pd.DataFrame([train_RMSE, test_RMSE]).T
        # summary.columns = ["Train RMSE", "Test RMSE"]
        # display(summary)
        print("Average Train / Test RMSE with Uncertainty:")
        print("(+3/+2): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_P32), stats.stdev(train_rmse_P32),
                 stats.mean(test_rmse_P32), stats.stdev(test_rmse_P32)))
        print("(+2/+1): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_P21), stats.stdev(train_rmse_P21),
                 stats.mean(test_rmse_P21), stats.stdev(test_rmse_P21)))
        print("(+1/0): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_P10), stats.stdev(train_rmse_P10),
                 stats.mean(test_rmse_P10), stats.stdev(test_rmse_P10)))
        print("(0/-1): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_N01), stats.stdev(train_rmse_N01),
                 stats.mean(test_rmse_N01), stats.stdev(test_rmse_N01)))
        print("(-1/-2): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_N12), stats.stdev(train_rmse_N12),
                 stats.mean(test_rmse_N12), stats.stdev(test_rmse_N12)))
        print("(-2/-3): %.3f +/- %.3f   /   %.3f +/- %.3f"
              % (stats.mean(train_rmse_N23), stats.stdev(train_rmse_N23),
                 stats.mean(test_rmse_N23), stats.stdev(test_rmse_N23)))
        print("")
        print("RMSE by defect site and SC type")
        print("---------------------------------------------------------")
        print("(+3/+2)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_P32),
                 stats.stdev(sub_test_rmse_P32)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_P32),
                 stats.stdev(int_test_rmse_P32)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_P32),
                 stats.stdev(IIVI_test_rmse_P32)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_P32),
                 stats.stdev(IIIV_test_rmse_P32)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_P32),
                 stats.stdev(IVIV_test_rmse_P32)))
        print("(+2/+1)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_P21),
                 stats.stdev(sub_test_rmse_P21)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_P21),
                 stats.stdev(int_test_rmse_P21)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_P21),
                 stats.stdev(IIVI_test_rmse_P21)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_P21),
                 stats.stdev(IIIV_test_rmse_P21)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_P21),
                 stats.stdev(IVIV_test_rmse_P21)))
        print("(+1/0)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_P10),
                 stats.stdev(sub_test_rmse_P10)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_P10),
                 stats.stdev(int_test_rmse_P10)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_P10),
                 stats.stdev(IIVI_test_rmse_P10)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_P10),
                 stats.stdev(IIIV_test_rmse_P10)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_P10),
                 stats.stdev(IVIV_test_rmse_P10)))
        print("(0/-1)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_N01),
                 stats.stdev(sub_test_rmse_N01)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_N01),
                 stats.stdev(int_test_rmse_N01)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_N01),
                 stats.stdev(IIVI_test_rmse_N01)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_N01),
                 stats.stdev(IIIV_test_rmse_N01)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_N01),
                 stats.stdev(IVIV_test_rmse_N01)))
        print("(-1/-2)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_N12),
                 stats.stdev(sub_test_rmse_N12)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_N12),
                 stats.stdev(int_test_rmse_N12)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_N12),
                 stats.stdev(IIVI_test_rmse_N12)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_N12),
                 stats.stdev(IIIV_test_rmse_N12)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_N12),
                 stats.stdev(IVIV_test_rmse_N12)))
        print("(-2/-3)")
        print("Sub site: %.3f +/- %.3f"
              % (stats.mean(sub_test_rmse_N23),
                 stats.stdev(sub_test_rmse_N23)))
        print("Int site: %.3f +/- %.3f"
              % (stats.mean(int_test_rmse_N23),
                 stats.stdev(int_test_rmse_N23)))
        print("IIVI type: %.3f +/- %.3f"
              % (stats.mean(IIVI_test_rmse_N23),
                 stats.stdev(IIVI_test_rmse_N23)))
        print("IIIV type: %.3f +/- %.3f"
              % (stats.mean(IIIV_test_rmse_N23),
                 stats.stdev(IIIV_test_rmse_N23)))
        print("IVIV type: %.3f +/- %.3f"
              % (stats.mean(IVIV_test_rmse_N23),
                 stats.stdev(IVIV_test_rmse_N23)))

        # plot error vs stdev
        fig, ax = plt.subplots(2, 3, figsize=(22, 10))
        fig.subplots_adjust(hspace=.25, wspace=0.175, top=.90)
        fig.suptitle("Transition Levels: Error vs. Uncertainty",
                     fontsize=20, va='top')
        fig.text(0.5, 0.05, 'DFT/NN Prediction Error (eV)', ha='center',
                 fontsize=17)
        fig.text(0.075, 0.5, 'STDEV - Uncertainty (eV)', va='center',
                 rotation='vertical', fontsize=17)

        subtitle_size = 17
        test_alpha = 0.65
        train_alpha = 0.85
        scatter_size = 6.5

        test_color = "#ff5042"   # red
        train_color = "#080bb6"  # blue

        # P32
        ax[0, 0].set_title("Transition Level: (+3/+2)", fontsize=subtitle_size)
        ax[0, 0].scatter(error_test_P32, std_test_P32, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[0, 0].scatter(error_train_P32, std_train_P32, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[0, 0].get_xlim(), ax[0, 0].get_ylim()]),
                np.max([ax[0, 0].get_xlim(), ax[0, 0].get_ylim()])]
        ax[0, 0].set_xlim(lims)
        ax[0, 0].set_ylim([0, np.amax([std_test_P32, std_train_P32])])

        # P21
        ax[1, 0].set_title("Transition Level: (+2/+1)", fontsize=subtitle_size)
        ax[1, 0].scatter(error_test_P21, std_test_P21, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[1, 0].scatter(error_train_P21, std_train_P21, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[1, 0].get_xlim(), ax[1, 0].get_ylim()]),
                np.max([ax[1, 0].get_xlim(), ax[1, 0].get_ylim()])]
        ax[1, 0].set_xlim(lims)
        ax[1, 0].set_ylim([0, np.amax([std_test_P21, std_train_P21])])

        # P10
        ax[0, 1].set_title("Transition Level: (+1/0)", fontsize=subtitle_size)
        ax[0, 1].scatter(error_test_P10, std_test_P10, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[0, 1].scatter(error_train_P10, std_train_P10, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[0, 1].get_xlim(), ax[0, 1].get_ylim()]),
                np.max([ax[0, 1].get_xlim(), ax[0, 1].get_ylim()])]
        ax[0, 1].set_xlim(lims)
        ax[0, 1].set_ylim([0, np.amax([std_test_P10, std_train_P10])])

        # N01
        ax[1, 1].set_title("Transition Level: (0/-1)", fontsize=subtitle_size)
        ax[1, 1].scatter(error_test_N01, std_test_N01, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[1, 1].scatter(error_train_N01, std_train_N01, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()]),
                np.max([ax[1, 1].get_xlim(), ax[1, 1].get_ylim()])]
        ax[1, 1].set_xlim(lims)
        ax[1, 1].set_ylim([0, np.amax([std_test_N01, std_train_N01])])

        # N12
        ax[0, 2].set_title("Transition Level: (-1/-2)", fontsize=subtitle_size)
        ax[0, 2].scatter(error_test_N12, std_test_N12, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[0, 2].scatter(error_train_N12, std_train_N12, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[0, 2].get_xlim(), ax[0, 2].get_ylim()]),
                np.max([ax[0, 2].get_xlim(), ax[0, 2].get_ylim()])]
        ax[0, 2].set_xlim(lims)
        ax[0, 2].set_ylim([0, np.amax([std_test_N12, std_train_N12])])

        # N23
        ax[1, 2].set_title("Transition Level: (-2/-3)", fontsize=subtitle_size)
        ax[1, 2].scatter(error_test_N23, std_test_N23, label="Test",
                         color=test_color, alpha=test_alpha, zorder=3,
                         s=scatter_size)
        ax[1, 2].scatter(error_train_N23, std_train_N23, label="Training",
                         color=train_color, alpha=train_alpha, zorder=1,
                         s=scatter_size)
        lims = [np.min([ax[1, 2].get_xlim(), ax[1, 2].get_ylim()]),
                np.max([ax[1, 2].get_xlim(), ax[1, 2].get_ylim()])]
        ax[1, 2].set_xlim(lims)
        ax[1, 2].set_ylim([0, np.amax([std_test_N23, std_train_N23])])

        return (error_train_P32, std_train_P32, error_test_P32, std_test_P32,
                error_train_P21, std_train_P21, error_test_P21, std_test_P21,
                error_train_P10, std_train_P10, error_test_P10, std_test_P10,
                error_train_N01, std_train_N01, error_test_N01, std_test_N01,
                error_train_N12, std_train_N12, error_test_N12, std_test_N12,
                error_train_N23, std_train_N23, error_test_N23, std_test_N23,
                ovr_predictions)

    def train_model(x_train_P32, x_train_P21, x_train_P10, x_train_N01,
                    x_train_N12, x_train_N23, y_train, x_test_P32, x_test_P21,
                    x_test_P10, x_test_N01, x_test_N12, x_test_N23, y_test,
                    in_dim_P32, in_dim_P21, in_dim_P10, in_dim_N01, in_dim_N12,
                    in_dim_N23, epochs, bs, lr, decay, dropout, noise, k_reg,
                    hid_layer_neurons, verbose, beta1, beta2, amsgrad):
        # Create inputs to model
        input_P32 = Input(shape=(in_dim_P32,))
        input_P21 = Input(shape=(in_dim_P21,))
        input_P10 = Input(shape=(in_dim_P10,))
        input_N01 = Input(shape=(in_dim_N01,))
        input_N12 = Input(shape=(in_dim_N12,))
        input_N23 = Input(shape=(in_dim_N23,))

        # P32
        P32 = Dense(round((in_dim_P32)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_P32)
        P32 = GaussianNoise(noise)(P32)
        P32 = GaussianDropout(dropout)(P32)
        P32 = Dense(round((in_dim_P32)/4), activation="relu",
                    activity_regularizer=l2(k_reg/10))(P32)
        P32 = Dense(round((in_dim_P32)/4), activation="relu",
                    activity_regularizer=l2(k_reg/10))(P32)

        # P21
        P21 = Dense(round((in_dim_P21)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_P21)
        P21 = GaussianNoise(noise)(P21)
        P21 = GaussianDropout(dropout)(P21)
        # P21 = Dense(round((in_dim_P21)/2)*2, activation="relu",
        #             activity_regularizer=l2(k_reg/10))(P21)
        # P21 = Dense(round((in_dim_P21)/2), activation="relu",
        #             activity_regularizer=l2(k_reg/10))(P21)

        # P10
        P10 = Dense(round((in_dim_P10)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_P10)
        P10 = GaussianNoise(noise)(P10)
        P10 = GaussianDropout(dropout)(P10)
        P10 = Dense(round((in_dim_P10)/2)*2, activation="relu",
                    activity_regularizer=l2(k_reg/10))(P10)
        # P10 = Dense(round((in_dim_P10)/2), activation="relu",
        #            activity_regularizer=l2(k_reg/10))(P10)

        # N01
        N01 = Dense(round((in_dim_N01)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_N01)
        N01 = GaussianNoise(noise)(N01)
        N01 = GaussianDropout(dropout)(N01)
        N01 = Dense(round((in_dim_N01)/2)*2, activation="relu",
                    activity_regularizer=l2(k_reg/10))(N01)
        # N01 = Dense(round((in_dim_N01)/2), activation="relu",
        #             activity_regularizer=l2(k_reg/10))(N01)

        # N12
        N12 = Dense(round((in_dim_N12)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_N12)
        N12 = GaussianNoise(noise)(N12)
        N12 = GaussianDropout(dropout)(N12)
        N12 = Dense(round((in_dim_N12)/2), activation="relu",
                    activity_regularizer=l2(k_reg/10))(N12)
        N12 = Dense(round((in_dim_N12)/2), activation="relu",
                    activity_regularizer=l2(k_reg/10))(N12)

        # N23
        N23 = Dense(round((in_dim_N23)/2)*2, activation="relu",
                    kernel_regularizer=l1(k_reg),
                    activity_regularizer=l2(k_reg/10))(input_N23)
        N23 = GaussianNoise(noise)(N23)
        N23 = GaussianDropout(dropout)(N23)
        N23 = Dense(round((in_dim_N23)/2), activation="relu",
                    activity_regularizer=l2(k_reg/10))(N23)
        N23 = Dense(round((in_dim_N23)/2), activation="relu",
                    activity_regularizer=l2(k_reg/10))(N23)

        # merge layers
        merge = concatenate([P32, P21, P10, N01, N12, N23])

        # Last Dense (Hidden) Layer
        hidden = Dense(hid_layer_neurons, activation="relu",
                       activity_regularizer=l2(k_reg/10))(merge)

        # output layer
        output = Dense(6)(hidden)

        # configure optimizer & compile model
        opt = Adam(lr=lr, beta_1=beta1, beta_2=beta2, decay=decay,
                   amsgrad=amsgrad)
        model = Model([input_P32, input_P21, input_P10, input_N01, input_N12,
                       input_N23], output)
        model.compile(loss="mse", optimizer=opt)

        # summarize model
        # print(model.summary())
        # plot_model(model, to_file='model_structure.png', show_shapes=True,
        #            show_layer_names=True)
        # display(Image(filename='model_structure.png'))

        # train model
        prediction = model.fit([x_train_P32, x_train_P21, x_train_P10,
                                x_train_N01, x_train_N12, x_train_N23],
                               y_train,
                               validation_data=([x_test_P32, x_test_P21,
                                                 x_test_P10, x_test_N01,
                                                 x_test_N12, x_test_N23],
                                                y_test),
                               epochs=epochs,
                               batch_size=bs,
                               verbose=verbose)

        return (model, prediction)
