# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python3
#     language: python
#     name: python3
# ---

import numpy as np
from functions import sigmoid

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return np.mean(e**2) / 2


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def calculate_logloss(y_true, y_pred, eps=1e-8):
    """Calculate the logloss"""
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )



def compute_loss(y, tx, w, loss_type):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        loss_type: string in ["mae", "mse", "log"] specifying the type of loss to compute

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y -tx.dot(w)

    if loss_type == "mse":
        return calculate_mse(e)

    elif loss_type == "mae":
        return calculate_mae(e)

    elif loss_type == "log":
        y_pred = sigmoid(tx @ w)
        return calculate_logloss(y, y_pred)

    else:
        raise ValueError(
            "Invalid value for argument 'loss_type' when calling compute_loss, 'type' must be in ['mse', 'mae', 'log']."
        )


