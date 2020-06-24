import sys
import pandas as pd
sys.path.append("../")
from neural_network import NeuralNetwork


class test_NeuralNetwork():
    def test_data_prep(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_graph_results(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_eval_catgr(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_model_eval(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_graph_prep(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_pred_fullchem(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_run_k_fold(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return

    def test_train_model(self):
        df = pd.read_csv("dummy_dft_df.csv", sep="|")
        assert NeuralNetwork.data_prep(df)
        return
