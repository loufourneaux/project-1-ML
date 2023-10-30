import numpy as np
from functions import sigmoid


def calculate_mse(e):
    """
    Calculate the mean squared error (MSE) for vector e.
    
    Args:
    e (numpy.ndarray): error vector
    
    Returns:
    float: mean squared error
    """
    return np.mean(e ** 2) / 2


def calculate_mae(e):
    """
    Calculate the mean absolute error (MAE) for vector e.
    
    Args:
    e (numpy.ndarray): error vector
    
    Returns:
    float: mean absolute error
    """
    return np.mean(np.abs(e))


def calculate_logloss(y_true, y_pred, eps=1e-8):
    """
    Calculate the logistic loss (logloss).
    
    Args:
    y_true (numpy.ndarray): true target values
    y_pred (numpy.ndarray): predicted target values
    eps (float, optional): a small number to prevent log(0)
    
    Returns:
    float: logistic loss
    """
    y_true = (y_true + 1) / 2  # Shift y_true from (-1, 1) to (0, 1)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def compute_loss(y, tx, w, loss_type):
    """
    Calculate the loss using either MSE, MAE, or logistic loss.
    
    Args:
    y (numpy.ndarray): target values
    tx (numpy.ndarray): input features
    w (numpy.ndarray): model weights
    loss_type (str): specifies the type of loss to compute ("mse", "mae", "log")
    
    Returns:
    float: loss value
    """
    y, w = y.reshape(-1, 1), w.reshape(-1, 1)
    e = y - tx.dot(w)

    if loss_type == "mse":
        return calculate_mse(e)
    elif loss_type == "mae":
        return calculate_mae(e)
    elif loss_type == "log":
        y_pred = sigmoid(tx @ w)
        return calculate_logloss(y, y_pred)
    else:
        raise ValueError(
            "Invalid value for argument 'loss_type'. Must be one of ['mse', 'mae', 'log']."
        )
