# -*- coding: utf-8 -*-
import numpy as np
from costs import *
from functions import *
from helpers import batch_iter


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm using MSE loss.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: loss mse value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "mse")

    for i in range(max_iters):

        # compute gradient
        grad = compute_gradient(y, tx, w)

        # update w by gradient descent
        w = w - gamma * grad

        # compute loss
        loss = compute_loss(y, tx, w, "mse")

        # Display current loss
        print("GD iter. {bi}/{ti}: loss={l}".format(bi=i, ti=max_iters - 1, l=loss))

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) using MSE loss.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: loss mse value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "mse")

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):

            # compute gradient
            grad = compute_gradient(minibatch_y, minibatch_tx, w)

            # update w through the stochastic gradient update
            w = w - gamma * grad

            # calculate loss
            loss = compute_loss(y, tx, w, "mse")

        # Display current loss
        print(
            "SGD iter. {bi}/{ti}: loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss)
        )

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
    returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: loss mse value (scalar)

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w, "mse")
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss mse value (scalar)
    """

    l = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx + l, tx.T @ y)
    loss = compute_loss(y, tx, w, "mse")
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using GD

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: log-loss value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "log")

    for i in range(max_iters):

        # compute gradient
        grad = compute_gradient(y, tx, w)

        # update w through the stochastic gradient update
        w = w - gamma * grad

        # calculate loss
        loss = compute_loss(y, tx, w, "log")

        # Display current loss
        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=i, ti=max_iters - 1, l=loss
            )
        )
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        lambda_: scalar.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: log-loss value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "log")

    for i in range(max_iters):

        # compute gradient
        grad = compute_gradient(y, tx, w)
        grad += 2 * lambda_ * w

        # update w through the stochastic gradient update
        w = w - gamma * grad

        # calculate loss
        loss = compute_loss(y, tx, w, "log")
        loss += lambda_ * np.linalg.norm(w) ** 2

        # Display current loss and weights
        print("GD iter. {bi}/{ti}: loss={l}".format(bi=i, ti=max_iters - 1, l=loss))
    return w, loss


