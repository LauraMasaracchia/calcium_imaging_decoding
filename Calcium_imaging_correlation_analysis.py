""" Analysis script that creates covariance or correlation matrices of the input,
 projects them to Riemann tangent space,
 Then runs Ridge Regression to decode any feature of interest.
 THIS FILES ASSUME YOU HAVE ALREADY PREPARES YOUR DATA IN THE FORM
 - calcium_imaging (np.array of shape = (n_trials, n_units, n_timepoints)) - trials should already be aligned at
                    meaningful point if necessary
 - labels (np.array of shape = (n_trials)) to decode - the variable to decode.
 """

import numpy as np
import os
import pickle
from Decoding_analysis.calcium_imaging_analysis.utils import create_sw_cov_or_corr_matrix
from Decoding_analysis.calcium_imaging_analysis.utils import project_to_tg_space
from Decoding_analysis.calcium_imaging_analysis.utils import GeneralizationAnalysisCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold


# -----------------------------------------------------------------------------------------------------------------
# ---------------------------------------- HYPERPARAMETERS -------------------------------------------------
# -------------- FILE NAMES AND FOLDERS ---------------------
data_name = "calcium_imaging_XXXXX.pkl"
# FOLDER WHERE THE DATA ARE - INTRODUCE FULL PATH
INPUT_FOLDER_NAME = "XXXXXXXX"
OUTPUT_FOLDER_NAME = "XXXXXX"

# ------------ ANALYSIS HYPERPARAMETERS--------------------
# DEFINE IF YOU WANT CORRELATION OR COVARIANCE MATRICES
COV_OR_CORR = 'covariance'
# DEFINE THE LENGTH OF SLIDING WINDOWS, in terms of data points
SLIDING_WINDOWS_SIZE = [6, 10, 20, 40]

# DEFINE CROSS VALIDATION NBR OF FOLDS
K_FOLD_NUMBER = 10
ALPHA = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

print("CREATE %s " % COV_OR_CORR + " matrices")
# define basename to use for saving the file
basename = COV_OR_CORR + "_mat_" + "session_XXXX"

# LOAD DATA
# !!!!!!!!!!!! PLEASE NOTE !!!!!!!!!!!!!!!!!!
# The file is supposed to contain two numpy arrays: calcium_imaging (shape = (n_trials,n_units,n_timepoints)),
# labels (shape = (n_trials)) to decode.
with open(os.path.join(INPUT_FOLDER_NAME, data_name), 'rb') as fp:
    calcium_imaging, labels = pickle.load(fp)

print('DATA loaded. Shape: ')
print(calcium_imaging.shape)
print(labels.shape)

n_trials, n_units, n_time_points = calcium_imaging.shape
n_windows = len(SLIDING_WINDOWS_SIZE)

# compute the covariance (or correlation) between units for every time point of the trial within the sliding window
cov_or_corr_matrices = {}
store_filename = basename + '-all_ws.pkl'
for ws in SLIDING_WINDOWS_SIZE:
    mat_name = '%s_ws%d' % (COV_OR_CORR, ws)
    cov_or_corr_matrices[mat_name] = create_sw_cov_or_corr_matrix(inputs=calcium_imaging,
                                                                  analysis_type=COV_OR_CORR,
                                                                  sliding_window_size=ws)

# store matrices, if needed
store_result_name = os.path.join(OUTPUT_FOLDER_NAME, store_filename)
with open(store_result_name, 'wb') as fp:
    pickle.dump((cov_or_corr_matrices, labels), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("%s matrices ready and stored" % COV_OR_CORR)

# LOAD MATRICES IF ALREADY CREATED
# print("ANALYSIS on INPUTS TYPE: %s " % COV_OR_CORR)
# with open(os.path.join(OUTPUT_FOLDER_NAME, store_filename), 'rb') as fp:
#     cov_or_corr_matrices, labels = pickle.load(fp)
#labels = labels.squeeze()


# optimization grid:
windows_general_matrix = np.zeros(shape=(n_windows, n_time_points, n_time_points))

kf = KFold(n_splits=K_FOLD_NUMBER, shuffle=True, random_state=10)
store_dict = {}
print("Analysis starts on %s matrices, all ws" % COV_OR_CORR)

for ws in range(n_windows):
    mat_name = '%s_ws%d' % (COV_OR_CORR, SLIDING_WINDOWS_SIZE[ws])
    mat_single_ws = cov_or_corr_matrices[mat_name]

    input_matrix = project_to_tg_space(mat_single_ws)
    print("tangent space projection ready for %s mat with ws %d" % (COV_OR_CORR, ws))

# -------------------------------------- ANALYSIS  ----------------------------------------------

    X = input_matrix
    y = labels

    # assume it's standard ridge on densities since there is another file for standard ridge on cov/corr
    model = RidgeClassifierCV()
    cross_validation_analysis = GeneralizationAnalysisCV(kf, model)
    accuracy_stored = cross_validation_analysis.run_analysis(X, y, K_FOLD_NUMBER)

    print("Analysis ready on window %d" % SLIDING_WINDOWS_SIZE[ws])
    windows_general_matrix[ws] = np.mean(accuracy_stored, axis=2)
# ------------------------------------------------------------------------------------------------------------

    store_dict['general_matrix_w%d' % SLIDING_WINDOWS_SIZE[ws]] = windows_general_matrix[ws]
    # store before going on new kernel


output_filename = "accuracy_ridge_" + store_filename
with open(os.path.join(OUTPUT_FOLDER_NAME, output_filename), 'wb') as fp:
    pickle.dump(store_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

print("Analysis finished. Accuracy matrices stored")

