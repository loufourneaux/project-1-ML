import numpy as np


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)
    e = y - np.dot(tx, w)
    gradient = (-1 / y.shape[0]) * np.dot(np.transpose(tx), e)
    return gradient


def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array"""
    return 1 / (1 + np.exp(-t))


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x.

    Args:
        x: numpy array of shape (N, D), where N is the number of samples, and D is the number of features.
        degree: integer, the degree of the polynomial basis.

    Returns:
        poly: numpy array of shape (N, D * (degree + 1)), representing the extended feature matrix.
    """

    N, D = x.shape
    poly = np.ones((N, D * (degree + 1)))

    for d in range(1, degree + 1):
        poly[:, D * d:D * (d + 1)] = x ** d

    return poly
