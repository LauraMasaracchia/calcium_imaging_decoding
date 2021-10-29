
""" In this file there are core functions and classes for the analysis"""
import numpy as np
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import zscore


class SlidingWindow(object):
    def __init__(self, window_size, data):
        self.ws = window_size
        self.half_w = int(self.ws/2)
        self.data = data
        self.list_length = self.data.shape[1]

    def slide_window(self, current_index):
        if current_index <= self.half_w:
            sel_data = self.data[:, 0:current_index + self.half_w]
        elif current_index >= self.list_length - self.half_w:
            sel_data = self.data[:, current_index - self.half_w:-1]
        else:
            sel_data = self.data[:, current_index-self.half_w: current_index+self.half_w]

        return sel_data


def create_sw_cov_or_corr_matrix(inputs, analysis_type, sliding_window_size):
    # mat_cov_all = np.zeros(shape=(n_windows, n_trials, n_units, n_units, n_time_points))
    n_trials, n_units, n_time_points = inputs.shape

    output_mat = np.zeros(shape=(n_trials, n_units, n_units, n_time_points))

    # loop on trials
    for k in range(n_trials):
        slider = SlidingWindow(sliding_window_size, inputs[k, :, :])

        for t in range(n_time_points):
            window_data = slider.slide_window(t)
            # mat_cov_all[s, k, :, :, t] = np.cov(window_data)
            if analysis_type == 'cov_mat':
                output_mat[k, :, :, t] = np.cov(window_data)
            elif analysis_type == 'corr_mat':
                output_mat[k, :, :, t] = np.corrcoef(window_data)

    return output_mat


def get_principal_components(X, DO_PCA, explained_variance=0.99):
    if DO_PCA:
        print("Running Principal Component Analysis on the input. New input shape:")
        pca = PCA(explained_variance)
        X = pca.fit_transform(X)
        print(X.shape)
        # Normalize after the PCA
        X = zscore(X)
    return X


def project_to_tg_space(cov_mat_single_ws):
    """ Function that uses the pyriemann library to project the input to their tangent space in Riemann Geometry.
    Used to reduce dimensions of input where euclidean distance is not necessarily meaningful,
    e.g. for the case of correlation/covariance"""
    n_time_points = cov_mat_single_ws.shape[3]
    n_units = cov_mat_single_ws.shape[1]
    n_trials = cov_mat_single_ws.shape[0]
    # create tangent space projection of the covariance matrices.
    ts = TangentSpace(metric='riemann')
    n_ts = int((n_units * (n_units + 1)) / 2)
    single_ws_input = np.zeros(shape=(n_trials, n_ts, n_time_points))
    # project every covariance matrix into tangent space
    for t_i in range(n_time_points):
        single_ws_input[:, :, t_i] = ts.fit_transform(cov_mat_single_ws[:, :, :, t_i])

    return single_ws_input


class GeneralizationAnalysisCV(object):
    """ Class initializing an analysis object.
    This analysis produces a temporal generalization matrix.
    It takes as input upon instantiation a kfold and a model pipeline.
    The analysis does nested cross validation on the model parameters inserted with the pipeline. """
    def __init__(self, kf, model_pipeline, accuracy_metric='accuracy_score'):
        self.model = model_pipeline
        self.kf = kf
        self.acc_metric = accuracy_metric

    def run_analysis(self, X, y, Kfold_nbr):
        """ Analysis runs: X assumed to be of the shape (n_trials, n_units, n_time_points).
        The analysis starts by dividing the trials in training and testing set (for every kfold).
        Then the model (model pipeline) is trained on a specific time point of the training set,
        and the same model is tested on all other time points of the test set.
        Accuracy matrices for every kfold are returned."""

        n_trials, n_units, n_time_points = X.shape

        accuracy_stored = np.zeros(shape=(n_time_points, n_time_points, Kfold_nbr))
        input_features = ['unit%d' % i for i in range(n_units)]
        indices = np.arange(n_trials)

        # loop on kfold splits
        k = 0
        for train_index, test_index in self.kf.split(indices):
            # split trials into train and test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #y_true_all = np.repeat(y_test, n_time_points)
            y_true_all = list(y_test) * n_time_points
            print(X_train.shape)
            print(y_train.shape)

            # train on one time point
            for t_i in range(n_time_points):
                X_train_timepoint = pd.DataFrame(X_train[:, :, t_i], columns=input_features)
                self.model.fit(X_train_timepoint, y_train)

                # concatenate all testing time points
                #X_test_tr = np.transpose(X_test, (0, 2, 1))
                X_test_tr = np.transpose(X_test, (2,0,1))
                X_test_tr = np.reshape(X_test_tr, (n_time_points * X_test.shape[0], n_units))
                X_test_tr_dataframe = pd.DataFrame(X_test_tr, columns=input_features)

                # test on all time points
                y_pred_all_time = self.model.predict(X_test_tr_dataframe)

                # Sign needed for the regression model with labels transformed to -1, 1,
                # and it does not change the output if the model is a classifier, with output 0, 1
                if self.acc_metric == "accuracy_score":
                    y_pred_all_time = np.sign(y_pred_all_time)
                    acc_all = y_true_all == y_pred_all_time * 1.0
                    #acc_reshape = np.reshape(acc_all, (y_test.shape[0], n_time_points))
                    #accuracy_stored[t_i, :, k] = np.mean(acc_reshape, axis=0)
                    acc_reshape = np.reshape(acc_all, (n_time_points, y_test.shape[0]))
                    accuracy_stored[t_i, :, k] = np.mean(acc_reshape, axis=1)
                elif self.acc_metric == "mean_squared_error":
                    acc_all = mean_squared_error(y_true_all, y_pred_all_time)
                    acc_reshape = np.reshape(acc_all, (y_test.shape[0], n_time_points))
                    accuracy_stored[t_i, :, k] = 1.- np.mean(acc_reshape, axis=0)

            print("mean overall accuracy fold %d" % k)
            print(np.mean(accuracy_stored[:, :, k]))
            print(np.var(accuracy_stored[:, :, k]))
            k += 1

        return accuracy_stored

