import numpy as np 
from implementationsoreo import *
from helpers import *
from functions import *
from costs import *
from data_processing import *

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
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1)
    w, loss_tr = mean_squared_error_gd(y_tr, xpoly_tr,ini_w,max_iters,gamma, divergence_ratio=1.01)
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



def cross_validation_logistic(y, x, k_indices, k,initial_w, max_iters, gamma, degree):
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
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1) 
    w, loss_tr = logistic_regression(y_tr, xpoly_tr,ini_w,max_iters,gamma)
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w,"mse"))
    return loss_tr, loss_te, w

def best_selection_logistic(y, x, max_iters, initial_ws_shape, gammas, degrees, k_fold, seed=1):
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
                    
                    _, loss_te,_ = cross_validation_logistic(y,x,k_indices, k, initial_w, max_iters, gamma, degree)
                    rmse_te_degree_gamma_w.append(loss_te)
                avg_rmse = np.mean(rmse_te_degree_gamma_w)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params['degree'] = degree
                    best_params['gamma'] = gamma
                    best_params['initial_w'] = initial_w

    
    return best_params
    
def cross_validation_reg_logistic(y, x, k_indices, k,initial_w, lamb, max_iters, gamma, degree):
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
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1) 
    w, loss_tr = reg_logistic_regression(y_tr, xpoly_tr,lamb,ini_w,max_iters,gamma)
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w,"mse"))
    return loss_tr, loss_te, w

def best_selection_reg_logistic(y, x, max_iters, initial_ws_shape, lambdas , gammas, degrees, k_fold, seed=1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    #for each degree, we compute the best lambdas and the associated rmse
    best_params = {'degree': None, 'gamma': None,'lambda':None ,'initial_w': None}
    best_rmse = float('inf')
    #vary degree
    for degree in degrees:
        # cross validation
        print('Currently trying : '+ str(degree))
        for gamma in gammas:
            print('Currently trying : '+ str(gamma))
            for l in lambdas:
                print('currently trying'+str(l))
                for initial_w in initial_ws_shape:

                    print('Currently trying : '+ str(initial_w))
                    rmse_te_degree_gamma_w_l =[]
                    for k in range(k_fold):
                    
                        _, loss_te,_ = cross_validation_reg_logistic(y,x,k_indices, k, l,initial_w, max_iters, gamma, degree)
                        rmse_te_degree_gamma_w_l.append(loss_te)
                    avg_rmse = np.mean(rmse_te_degree_gamma_w_l)
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params['degree'] = degree
                        best_params['gamma'] = gamma
                        best_params['initial_w'] = initial_w
                        best_params['lambda']=l
    return best_params

def cross_validation_least_squares(y, x, k_indices, k, degree):
    """return the loss of least squares regression"""
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
    w, loss_tr = least_squares(y_tr, xpoly_tr) 
    # calculate the loss for test data:
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))
    return loss_tr, loss_te, w

def best_degree_selection_least_squares(y, x, degrees, k_fold, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #vary degree
    rmse_te=[]
    for degree in degrees:
        rmse_te_tmp= []
        for k in range(k_fold):
            _, loss_te,_ = cross_validation_least_squares(y, x, k_indices, k,degree)
            rmse_te_tmp.append(loss_te)
        rmse_te.append(np.mean(rmse_te_tmp))
    
         
    ind_best_degree =  np.argmin(rmse_te)      
        
    return degrees[ind_best_degree]
