# Importing necessary libraries and modules
from implementations import *
from functions import *
from costs import *
from data_processing import *


# This function is used to build k indices for k-fold cross-validation
def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    Args:
    y (numpy.ndarray): target values, shape=(N,)
    k_fold (int): number of folds
    seed (int): random seed

    Returns:
    numpy.ndarray: A 2D array of shape=(k_fold, N/k_fold) indicating the data indices for each fold
    """

    num_row = y.shape[0]  # Number of rows/data points
    interval = int(num_row / k_fold)  # Interval to split the data
    np.random.seed(seed)  # Setting random seed
    indices = np.random.permutation(num_row)  # Permuting the indices randomly
    # Splitting the indices for k folds
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)  # Returning a numpy array of indices


# This function performs cross-validation for ridge regression
def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """
    Return the loss of ridge regression.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): k_indices from the build_k_indices function
    k (int): specifies the kth fold to be used for validation
    lambda_ (float): regularization parameter
    degree (int): degree of polynomial expansion for the features
    
    Returns:
    float, float, numpy.ndarray: training loss, test loss, weights of the ridge regression model
    """

    # Getting the indices for test and train data
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)

    # Splitting the data into test and train sets
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]

    # Building polynomial features
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)

    # Calculating weights and training loss for ridge regression
    w, loss_tr = ridge_regression(y_tr, xpoly_tr, lambda_)

    # Calculating the loss for test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))

    return loss_tr, loss_te, w  # Returning the training and test losses, and the weights


def best_selection_ridge(y, x, degrees, k_fold, lambdas, seed=1):
    """
    Selects the best hyperparameters (degree and lambda) for ridge regression using cross-validation.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list): degrees to be considered for polynomial expansion
    k_fold (int): number of folds for cross-validation
    lambdas (list): list of lambda values (regularization parameter) to be tested
    seed (int, optional): random seed for splitting the data, default is 1
    
    Returns:
    dict: dictionary containing the best degree and lambda values
    """

    # Splitting data into k folds
    k_indices = build_k_indices(y, k_fold, seed)

    # Initial best parameters and error values
    best_params = {'degree': None, 'lambda': None}
    best_rmse = float('inf')  # Set to infinity initially for comparison

    # Iterating over each degree
    for degree in degrees:
        # Performing cross-validation for each lambda
        for lambda_ in lambdas:
            rmse_te_degree_lambda = []  # List to store test errors for each fold

            # Performing k-fold cross-validation
            for k in range(k_fold):
                _, loss_te, _ = cross_validation_ridge(y, x, k_indices, k, lambda_, degree)
                rmse_te_degree_lambda.append(loss_te)  # Storing test errors

            # Calculating average test error across all folds
            avg_rmse = np.mean(rmse_te_degree_lambda)

            # Updating best parameters if a lower error is found
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params['degree'] = degree
                best_params['lambda'] = lambda_

    return best_params  # Returning the best hyperparameters


def cross_validation_gd(y, x, k_indices, k, initial_w, max_iters, gamma, degree):
    """
    Perform k-fold cross-validation for gradient descent and returns the training and test loss.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): k indices for k-fold
    k (int): the kth fold
    initial_w (str): initialization strategy for weights ('ones', 'zeros', 'random')
    max_iters (int): max number of iterations for gradient descent
    gamma (float): learning rate
    degree (int): polynomial degree for feature expansion
    
    Returns:
    float: training loss
    float: test loss
    numpy.ndarray: weights vector
    """

    # Get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)

    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]

    # Form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)

    # Initialize weights
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1)

    # Compute weights and training loss for gradient descent model
    w, loss_tr = mean_squared_error_gd(y_tr, xpoly_tr, ini_w, max_iters, gamma, divergence_ratio=1.01)

    # Calculate the loss for test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))

    return loss_tr, loss_te, w  # Return training loss, test loss, and weights


def best_selection_gd(y, x, degrees, k_fold, initial_ws, max_iters, gammas, seed=1):
    """
    Perform hyperparameter tuning for gradient descent using k-fold cross-validation. 
    Finds the best polynomial degree, learning rate (gamma), and weights initialization.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list): degrees to be tested
    k_fold (int): number of folds
    initial_ws (list): list of weights initialization strategies
    max_iters (int): max number of iterations
    gammas (list): learning rates to be tested
    seed (int): random seed for reproducibility
    
    Returns:
    dict: dictionary containing the best hyperparameters
    """

    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_params = {'degree': None, 'gamma': None, 'initial_w': None}
    best_rmse = float('inf')

    # Vary degree
    for degree in degrees:

        # Cross-validation
        for gamma in gammas:
            for initial_w in initial_ws:
                rmse_te_degree_gamma_w = []

                for k in range(k_fold):
                    _, loss_te, _ = cross_validation_gd(y, x, k_indices, k, initial_w, max_iters, gamma, degree)
                    rmse_te_degree_gamma_w.append(loss_te)

                avg_rmse = np.mean(rmse_te_degree_gamma_w)

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params['degree'] = degree
                    best_params['gamma'] = gamma
                    best_params['initial_w'] = initial_w

    return best_params  # Return the best found hyperparameters


def cross_validation_sgd(y, x, k_indices, k, initial_w, max_iters, gamma, degree):
    """
    Perform k-fold cross-validation for Stochastic Gradient Descent (SGD).
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): k indices for the k-fold
    k (int): current fold index
    initial_w (str): weight initialization strategy ('ones', 'zeros', 'random')
    max_iters (int): maximum number of iterations for SGD
    gamma (float): learning rate
    degree (int): polynomial degree for feature expansion
    
    Returns:
    float, float, numpy.ndarray: training loss, test loss, and weights
    """

    # Get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)
    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]

    # Form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    # Weights initialization
    if initial_w == 'ones':
        ini_w = np.ones((tx_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((tx_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(tx_tr.shape[1], 1)

    # Apply SGD and calculate training loss
    w, loss_tr = mean_squared_error_sgd(y_tr, tx_tr, ini_w, max_iters, gamma)

    # Calculate the loss for test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w, "mse"))

    return loss_tr, loss_te, w  # Return training and test loss, and weights


def best_selection_sgd(y, x, degrees, k_fold, initial_ws, max_iters, gammas, seed=1):
    """
    Find the best hyperparameters (degree, gamma, initial_w) for SGD using k-fold CV.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list): degrees to be tested
    k_fold (int): number of folds
    initial_ws (list): initial weights options to be tested
    max_iters (int): maximum number of iterations
    gammas (list): learning rates to be tested
    seed (int, optional): random seed
    
    Returns:
    dict: best parameters found
    """

    # Split data in k folds
    k_indices = build_k_indices(y, k_fold, seed)

    best_params = {'degree': None, 'gamma': None, 'initial_w': None}
    best_rmse = float('inf')

    # Vary degree
    for degree in degrees:
        # Cross validation
        for gamma in gammas:
            for initial_w in initial_ws:
                rmse_te = []  # Store test RMSE for each fold

                for k in range(k_fold):
                    _, loss_te, _ = cross_validation_sgd(y, x, k_indices, k, initial_w, max_iters, gamma, degree)
                    rmse_te.append(loss_te)

                avg_rmse = np.mean(rmse_te)  # Average test RMSE over all folds

                # Update best parameters if current configuration is better
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params['degree'] = degree
                    best_params['gamma'] = gamma
                    best_params['initial_w'] = initial_w

    return best_params  # Return the best parameters found


def cross_validation_logistic(y, x, k_indices, k, initial_w, max_iters, gamma, degree):
    """
    Perform k-fold cross validation for logistic regression.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): k indices for k-fold
    k (int): current fold index
    initial_w (str): initial weight strategy ('ones', 'zeros', 'random')
    max_iters (int): maximum number of iterations
    gamma (float): learning rate
    degree (int): polynomial degree
    
    Returns:
    float, float, numpy.ndarray: training loss, test loss, weights
    """

    # Get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)

    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]

    # Form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)

    # Initialize weights
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1)

    # Perform logistic regression
    w, loss_tr = logistic_regression(y_tr, xpoly_tr, ini_w, max_iters, gamma)

    # Calculate the loss for test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))

    return loss_tr, loss_te, w  # Return training and test losses, and weights


def best_selection_logistic(y, x, degrees, k_fold, initial_ws_shape, max_iters, gammas, seed=1):
    """
    Select the best hyperparameters for logistic regression using k-fold cross-validation.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list): polynomial degrees to test
    k_fold (int): number of folds
    initial_ws_shape (list): initial weight strategies to test ('ones', 'zeros', 'random')
    max_iters (int): maximum number of iterations
    gammas (list): learning rates to test
    seed (int): random seed
    
    Returns:
    dict: best hyperparameters
    """

    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Initialize best parameters
    best_params = {'degree': None, 'gamma': None, 'initial_w': None}
    best_rmse = float('inf')

    # Vary degree, learning rate (gamma), and initial weights
    for degree in degrees:
        for gamma in gammas:
            for initial_w in initial_ws_shape:
                rmse_te_degree_gamma_w = []
                for k in range(k_fold):
                    # Perform cross-validation
                    _, loss_te, _ = cross_validation_logistic(y, x, k_indices, k, initial_w, max_iters, gamma, degree)
                    rmse_te_degree_gamma_w.append(loss_te)
                # Average RMSE
                avg_rmse = np.mean(rmse_te_degree_gamma_w)
                if avg_rmse < best_rmse:
                    # Update best parameters if current RMSE is lower
                    best_rmse = avg_rmse
                    best_params['degree'] = degree
                    best_params['gamma'] = gamma
                    best_params['initial_w'] = initial_w

    return best_params  # Return the best found hyperparameters


def cross_validation_reg_logistic(y, x, k_indices, k, initial_w, lamb, max_iters, gamma, degree):
    """
    Perform k-fold cross-validation for regularized logistic regression.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): indices for k-fold cross-validation
    k (int): current fold index
    initial_w (str): strategy for initializing weights ('ones', 'zeros', 'random')
    lamb (float): regularization parameter
    max_iters (int): maximum number of iterations
    gamma (float): learning rate
    degree (int): polynomial degree
    
    Returns:
    float: training loss
    float: test loss
    numpy.ndarray: weights
    """

    # Get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_index = tr_index.reshape(-1)

    y_te = y[te_index]
    y_tr = y[tr_index]
    x_te = x[te_index]
    x_tr = x[tr_index]

    # Form data with polynomial degree
    xpoly_tr = build_poly(x_tr, degree)
    xpoly_te = build_poly(x_te, degree)

    # Initialize weights
    if initial_w == 'ones':
        ini_w = np.ones((xpoly_tr.shape[1], 1))
    elif initial_w == 'zeros':
        ini_w = np.zeros((xpoly_tr.shape[1], 1))
    elif initial_w == 'random':
        np.random.seed(42)
        ini_w = np.random.rand(xpoly_tr.shape[1], 1)

    # Train using regularized logistic regression
    w, loss_tr = reg_logistic_regression(y_tr, xpoly_tr, lamb, ini_w, max_iters, gamma)

    # Calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))

    return loss_tr, loss_te, w  # Return training and test loss, and weights


def best_selection_reg_logistic(y, x, degrees, k_fold, max_iters, initial_ws_shape, lambdas, gammas, seed=1):
    """
    Select the best hyperparameters for regularized logistic regression using cross-validation.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list of int): degrees for polynomial expansion
    k_fold (int): number of folds
    max_iters (int): maximum number of iterations
    initial_ws_shape (list of str): initialization strategies for weights
    lambdas (list of float): regularization parameters
    gammas (list of float): learning rates
    seed (int, optional): random seed
    
    Returns:
    dict: best hyperparameters
    """

    # Split data into k folds
    k_indices = build_k_indices(y, k_fold, seed)

    # Initialize best parameters and RMSE
    best_params = {'degree': None, 'gamma': None, 'lambda': None, 'initial_w': None}
    best_rmse = float('inf')

    # Vary hyperparameters to find the best combination
    for degree in degrees:
        for gamma in gammas:
            for lamb in lambdas:
                for initial_w in initial_ws_shape:
                    rmse_te_list = []  # To store test RMSE for each fold
                    for k in range(k_fold):
                        _, loss_te, _ = cross_validation_reg_logistic(y, x, k_indices, k, initial_w, lamb, max_iters,
                                                                      gamma, degree)
                        rmse_te_list.append(loss_te)

                    # Calculate average test RMSE across folds
                    avg_rmse = np.mean(rmse_te_list)

                    # Update best parameters and RMSE if a better combination is found
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = {'degree': degree, 'gamma': gamma, 'lambda': lamb, 'initial_w': initial_w}

    return best_params  # Return best hyperparameters


def cross_validation_least_squares(y, x, k_indices, k, degree):
    """
    Perform k-fold cross-validation on least squares regression.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    k_indices (numpy.ndarray): k different group indices
    k (int): index for the test group
    degree (int): polynomial degree
    
    Returns:
    tuple: training loss, test loss, weights
    """
    # Get k'th subgroup in test, others in train
    te_index = k_indices[k]
    tr_index = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    y_te, y_tr = y[te_index], y[tr_index]
    x_te, x_tr = x[te_index], x[tr_index]

    # Form data with polynomial degree
    xpoly_tr, xpoly_te = build_poly(x_tr, degree), build_poly(x_te, degree)

    # Weights and training loss for least squares model
    w, loss_tr = least_squares(y_tr, xpoly_tr)

    # Calculate the loss for test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, xpoly_tr, w, "mse"))
    loss_te = np.sqrt(2 * compute_loss(y_te, xpoly_te, w, "mse"))

    return loss_tr, loss_te, w


def best_degree_selection_least_squares(y, x, degrees, k_fold, seed=1):
    """
    Find the best polynomial degree for least squares regression using k-fold cross-validation.
    
    Args:
    y (numpy.ndarray): target values
    x (numpy.ndarray): input features
    degrees (list of int): degrees for polynomial expansion
    k_fold (int): number of folds
    seed (int, optional): random seed
    
    Returns:
    int: best degree
    """
    # Split data into k folds
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_te = []
    # Vary degree to find the best one
    for degree in degrees:
        rmse_te_tmp = []
        for k in range(k_fold):
            _, loss_te, _ = cross_validation_least_squares(y, x, k_indices, k, degree)
            rmse_te_tmp.append(loss_te)
        rmse_te.append(np.mean(rmse_te_tmp))

    # Find the degree that gives the lowest average test RMSE
    ind_best_degree = np.argmin(rmse_te)

    return degrees[ind_best_degree]
