import numpy as np
from helpers import *
from data_processing import *
from cross_validation import *

x_train, x_test, y_train, train_ids, test_ids = load_csv_data('.\\dataset')
x_train_preprocessed = clean_all(x_train)
x_test_preprocessed = clean_all(x_test)
x_tr, x_te, y_tr, y_te = split_data(x_train_preprocessed, y_train, 0.95)
f1_regression_score = {'GD': None,
                       'SGD': None,
                       'LS': None,
                       'Ridge': None,
                       'Log': None,
                       'RegLog': None}

best_parameters_per_regression_model = {'GD': None,
                                        'SGD': None,
                                        'LS': None,
                                        'Ridge': None,
                                        'Log': None,
                                        'RegLog': None}
initial_ws = []
initial_ws.append('random')
k_fold = 5
max_iter = 1000
gammas = np.arange(0.005, 0.016, 0.001)
gammas = np.round(gammas, 3)
degrees = np.arange(4)
lambdas_ridge = np.arange(0.01, 0.1, 0.05)
lambdas_ridge = np.round(lambdas_ridge, 2)
lambdas = np.arange(0.0001, 0.001, 0.005)
lambdas = np.round(lambdas, 4)
best_parameters_per_regression_model['GD'] = best_selection_gd(y_tr, x_tr, [degrees[0]], k_fold, initial_ws, max_iter,
                                                               gammas)
best_parameters_per_regression_model['SGD'] = best_selection_sgd(y_tr, x_tr, [degrees[0]], k_fold, initial_ws, max_iter,
                                                                 gammas)
best_parameters_per_regression_model['Ridge'] = best_selection_ridge(y_tr, x_tr, [degrees[0]], k_fold, lambdas_ridge)
best_parameters_per_regression_model['Log'] = best_selection_logistic(y_tr, x_tr, [degrees[0]], k_fold, initial_ws,
                                                                      max_iter, gammas)
best_parameters_per_regression_model['RegLog'] = best_selection_reg_logistic(y_tr, x_tr, degrees, k_fold, max_iter,
                                                                             initial_ws, lambdas, gammas)
np.random.seed(42)
ini_w = np.random.rand(x_tr.shape[1], 1)
w_gd, loss_gd = mean_squared_error_gd(y_train, x_train_preprocessed, ini_w, max_iter,
                                      best_parameters_per_regression_model['GD']['gamma'])
w_sgd, loss_sgd = mean_squared_error_sgd(y_train, x_train_preprocessed, ini_w, max_iter,
                                         best_parameters_per_regression_model['SGD']['gamma'])
w_ls, loss_ls = least_squares(y_train, x_train_preprocessed)
w_ridge, loss_ridge = ridge_regression(y_train, x_train_preprocessed,
                                       best_parameters_per_regression_model['Ridge']['lambda'])
w_log, loss_log = logistic_regression(y_train, x_train_preprocessed, ini_w, max_iter,
                                      best_parameters_per_regression_model['Log']['gamma'])
w_reglog, loss_reglog = reg_logistic_regression(y_train, x_train_preprocessed,
                                                best_parameters_per_regression_model['RegLog']['lambda'], ini_w,
                                                max_iter, best_parameters_per_regression_model['RegLog']['gamma'])
f1_regression_score['GD'] = compute_f1_score(y_te, prediction(x_te, w_gd))
f1_regression_score['SGD'] = compute_f1_score(y_te, prediction(x_te, w_sgd))
f1_regression_score['LS'] = compute_f1_score(y_te, prediction(x_te, w_ls))
f1_regression_score['Ridge'] = compute_f1_score(y_te, prediction(x_te, w_ridge))
f1_regression_score['Log'] = compute_f1_score(y_te, prediction(x_te, w_log))
f1_regression_score['RegLog'] = compute_f1_score(y_te, prediction(x_te, w_reglog))
best_score = -1
best_method = None

for method, score in f1_regression_score.items():
    if score is not None:
        if score > best_score:
            best_score = score
            best_method = method

w_pred = 0
loss_pred = 0
if best_method == 'GD':
    w_pred = w_gd
if best_method == 'SGD':
    w_pred = w_sgd
if best_method == 'LS':
    w_pred = w_ls
if best_method == 'Ridge':
    w_pred = w_ridge
if best_method == 'Log':
    w_pred = w_log
if best_method == 'Reglog':
    w_pred = w_reglog
y_pred = prediction(x_test_preprocessed, w_pred)
create_csv_submission(test_ids, y_pred, 'submission')

#%%
