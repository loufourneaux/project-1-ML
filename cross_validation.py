import numpy as np 
from implementationsoreo import *
from helpers import *
from functions import *
from costs import *
from data_processing import *


def cross_validation_with_degree(y, tx, k_fold, hyperparameters, method="ridge"):
    """
    Perform cross-validation to tune hyperparameters for different regression methods, including degree.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        k_fold: Number of cross-validation folds
        hyperparameters: Dictionary containing hyperparameters specific to the selected method,
            including "degree" for polynomial regression methods.
        method: string, the name of the regression method ("ridge", "least_squares", "mean_squared_error_gd", "mean_squared_error_sgd", "logistic", or "reg_logistic").

    Returns:
        best_hyperparameters: Dictionary with the best hyperparameters
        best_loss: The lowest loss obtained during cross-validation
    """
    num_samples = len(y)
    fold_size = num_samples // k_fold
    best_loss = float("inf")
    best_hyperparameters = {}

    for hyperparameter_set in hyperparameters:
        total_loss = 0.0

        for fold in range(k_fold):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            val_indices = np.arange(val_start, val_end)
            train_indices = np.delete(np.arange(num_samples), val_indices)

            y_train, tx_train = y[train_indices], tx[train_indices]
            y_val, tx_val = y[val_indices], tx[val_indices]

            if "degree" in hyperparameter_set:
                degree = hyperparameter_set["degree"]
                tx_train = build_poly(tx_train, degree)
                tx_val = build_poly(tx_val, degree)

            if method == "ridge":
                lambda_ = hyperparameter_set["lambda"]
                w = ridge_regression(y_train, tx_train, lambda_)
            elif method == "least_squares":
                w, _ = least_squares(y_train, tx_train)
            elif method == "mean_squared_error_gd":
                initial_w = hyperparameter_set["initial_w"]
                max_iters = hyperparameter_set["max_iters"]
                gamma = hyperparameter_set["gamma"]
                w, _ = mean_squared_error_gd(y_train, tx_train, initial_w, max_iters, gamma)
            elif method == "mean_squared_error_sgd":
                initial_w = hyperparameter_set["initial_w"]
                max_iters = hyperparameter_set["max_iters"]
                gamma = hyperparameter_set["gamma"]
                batch_size = hyperparameter_set["batch_size"]
                w, _ = mean_squared_error_sgd(y_train, tx_train, initial_w, max_iters, gamma)#, batch_size)
            elif method == "logistic":
                initial_w = hyperparameter_set["initial_w"]
                max_iters = hyperparameter_set["max_iters"]
                gamma = hyperparameter_set["gamma"]
                w, _ = logistic_regression(y_train, tx_train, initial_w, max_iters, gamma)
            elif method == "reg_logistic":
                lambda_ = hyperparameter_set["lambda"]
                initial_w = hyperparameter_set["initial_w"]
                max_iters = hyperparameter_set["max_iters"]
                gamma = hyperparameter_set["gamma"]
                w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
            else:
                raise ValueError("Invalid regression method.")

            if method in ["ridge", "logistic", "reg_logistic"]:
                val_loss = compute_loss(y_val, tx_val, w, "log")
            else:
                val_loss = np.sqrt(2 * compute_loss(y_val, tx_val, w, "mse"))

            total_loss += val_loss

        average_loss = total_loss / k_fold

        if average_loss < best_loss:
            best_loss = average_loss
            best_hyperparameters = hyperparameter_set

    return best_hyperparameters, best_loss

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]
    # form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)
    # weights and training loss for ridge regression model:
    w, loss_tr = ridge_regression(y_tr, xpoly_tr, lambda_) 
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w,"mse"))
    return loss_tr, loss_te, w

def best_degree_selection_ridge(y, x, degrees, k_fold, lambdas, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te,_ = cross_validation_ridge(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree], lambdas[ind_lambda_opt]


def cross_validation_logistic(y, x, k_indices, k, max_iters, gamma, degree):
    """return the loss of log regression."""
    # get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]
    # form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)
    #initialize weight vector:
    initial_w = np.random.normal(0., 0.1, [xpoly_tr.shape[1],])
    # weights and training loss for ridge regression model:
    w, loss_tr = logistic_regression(y_tr, xpoly_tr, initial_w, max_iters, gamma) 
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "log"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "log"))
    return loss_tr, loss_te, w

def best_degree_selection_logistic(y, x, max_iters, gamma, degrees, k_fold, seed=1):
    #split data in k fold:
    k_indices = build_k_indices(y, k_fold, seed)
    
    for degree in degrees:
        losses_te = []
        for k in range(k_fold):
            _, loss_te,_ = cross_validation_logistic(y, x, k_indices, k, max_iters, gamma, degree)
            losses_te.append(loss_te)
            
    ind_degree_opt = np.argmin(losses_te)
    
    return degrees[ind_degree_opt]

def cross_validation_gd(y, x, k_indices, k, initial_w, max_iters, gamma, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]
    # form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)
    # weights and training loss for gradient descent model:
    print(y.shape)
    print(xpoly_tr.shape)
    ini_w = np.ones((xpoly_tr.shape[1], 1)) if initial_w == 'ones' else np.zeros((xpoly_tr.shape[1], 1))
    w, loss_tr = mean_squared_error_gd(y_tr, xpoly_tr,ini_w,max_iters,gamma)
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w,"mse"))
    return loss_tr, loss_te, w
    

def best_selection_gd(y, x, degrees, k_fold, initial_ws_shape, max_iters,gammas, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    #for each degree, we compute the best lambdas and the associated rmse
    best_params = {'degree': None, 'gamma': None, 'initial_w': None}
    best_rmse = float('inf')
    #vary degree
    for degree in degrees:
        # cross validation
        print('Currently trying : '+ str(degree))
        for gamma in gammas:
            print('Currently trying : '+ str(gamma))
            
            for initial_w in initial_ws_shape:

                print('Currently trying : '+ str(initial_w))
                rmse_te_degree_gamma_w =[]
                for k in range(k_fold):
                    
                    _, loss_te,_ = cross_validation_gd(y,x,k_indices, k, initial_w, max_iters, gamma, degree)
                    rmse_te_degree_gamma_w.append(loss_te)
                avg_rmse = np.mean(rmse_te_degree_gamma_w)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params['degree'] = degree
                    best_params['gamma'] = gamma
                    best_params['initial_w'] = initial_w

        
    
    return best_params


