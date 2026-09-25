import numpy as np

def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx@w 
    loss = 1/(2*y.shape[0])*np.sum(e**2)

    return loss

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - tx@w
    gradient = -1/y.shape[0] * tx.T@e
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma) : 
    """The Gradient Descent (GD) algorithm.
    
        Args:
            y: shape=(N, )
            tx: shape=(N,2)
            initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of GD
            gamma: a scalar denoting the stepsize
    
        Returns:
            (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
            
        """

    w = initial_w

    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx ,w)
        gradient = compute_gradient(y, tx, w)
        w = w-gamma*gradient

    return (w, loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma) :
    raise NotImplementedError

def least_squares(y, tx) : 
    raise NotImplementedError

def ridge_regression(y, tx, lambda_) :
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) :
    raise NotImplementedError


