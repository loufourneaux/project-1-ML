from functions import *
from helpers import batch_iter
from costs import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, tol=1e-5, divergence_ratio=1.5):
    """The Gradient Descent (GD) algorithm using MSE loss.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        tol: tolerance for convergence criteria
        divergence_ratio: ratio to consider the loss as diverging

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: loss mse value (scalar)
    """

    w = initial_w
    loss = compute_loss(y, tx, w, "mse")
    i = 0
    loss_change = tol + 1  # Initial loss change to enter the loop

    while i < max_iters and loss_change > tol:

        prev_loss = loss  # Save previous loss value

        # compute gradient
        grad = compute_gradient(y, tx, w)

        # update w by gradient descent
        w = w - gamma * grad

        # compute loss
        loss = compute_loss(y, tx, w, "mse")

        # Calculate the change in the loss function
        loss_change = abs(prev_loss - loss)

        if loss > divergence_ratio * prev_loss:
            print("Divergence detected. Stopping iteration.")
            break

        i += 1

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, tol=1e-6, divergence_ratio=1.01):
    """The Stochastic Gradient Descent algorithm (SGD) using MSE loss.

     Args:
        y (array): numpy array of shape=(N, ) representing target values
        tx (array): numpy array of shape=(N,D) representing feature values
        initial_w (array): numpy array of shape=(D, ). The initial guess for the model parameters
        max_iters (int): a scalar denoting the total number of iterations of SGD
        gamma (float): a scalar denoting the stepsize
        tol (float, optional): Tolerance for the difference in loss. Defaults to 1e-6.
        divergence_ratio (float, optional): Ratio to determine divergence. Defaults to 1.01.

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: loss mse value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "mse")
    i = 0
    loss_change = tol + 1  # Initial loss change to enter the loop
    did_break = False
    while i < max_iters and loss_change > tol:
        if did_break:
            break
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            prev_loss = loss  # Save previous loss value

            # compute gradient
            grad = compute_gradient(minibatch_y, minibatch_tx, w)

            # update w through the stochastic gradient update
            w = w - gamma * grad

            # calculate loss
            loss = compute_loss(y, tx, w, "mse")

            if loss > divergence_ratio * prev_loss:
                print("Divergence detected. Stopping iteration.")
                did_break = True
                break

            # Calculate the change in the loss function
            loss_change = abs(prev_loss - loss)
            # Display current loss
        i += 1
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


def logistic_regression(y, tx, initial_w, max_iters, gamma, tol=1e-6, divergence_ratio=1.01):
    """Logistic regression using GD

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        tol (float, optional): Tolerance for the difference in loss. Defaults to 1e-6.
        divergence_ratio (float, optional): Ratio to determine divergence. Defaults to 1.01.

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: log-loss value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "log")
    i = 0
    loss_change = tol + 1  # Initial loss change to enter the loop
    while i < max_iters and loss_change > tol:

        prev_loss = loss  # Save previous loss value

        # compute gradient
        grad = compute_gradient(y, tx, w)

        # update w by gradient descent
        w -= gamma * grad

        # compute loss
        loss = compute_loss(y, tx, w, "log")

        # Calculate the change in the loss function
        loss_change = abs(prev_loss - loss)

        if loss > divergence_ratio * prev_loss:
            print("Divergence detected. Stopping iteration.")
            break

        i += 1

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, tol=1e-6, divergence_ratio=1.01):
    """Regularized logistic regression using GD

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        lambda_: scalar.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        tol (float, optional): Tolerance for the difference in loss. Defaults to 1e-6.
        divergence_ratio (float, optional): Ratio to determine divergence. Defaults to 1.01.

    Returns:
        w: model parameters as numpy arrays of shape (D, )
        loss: log-loss value (scalar)
    """

    # Initialize weights and loss
    w = initial_w
    loss = compute_loss(y, tx, w, "log")
    i = 0
    loss_change = tol + 1  # Initial loss change to enter the loop
    while i < max_iters and loss_change > tol:

        prev_loss = loss  # Save previous loss value

        # compute gradient
        grad = compute_gradient(y, tx, w)
        grad += 2 * lambda_ * w

        # update w by gradient descent
        w -= gamma * grad

        # compute loss
        loss = compute_loss(y, tx, w, "log")
        loss += lambda_ * (np.linalg.norm(w) ** 2)

        # Calculate the change in the loss function
        loss_change = abs(prev_loss - loss)

        if loss > divergence_ratio * prev_loss:
            print("Divergence detected. Stopping iteration.")
            break

    return w, loss
