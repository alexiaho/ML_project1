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
    e = y - tx @ w 
    return 1 / 2 * np.mean(e**2)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    gradient = -1/y.shape[0] * tx.T @ e
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
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    
    loss = compute_loss_mse(y, tx ,w) # Loss computed after final weight update
    
    return (w, loss)
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma) :
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    """
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            
            # Compute a stochastic gradient and loss
            gradient = compute_gradient(y_batch, tx_batch, w)  # gradient computed on small batch
            
            # Update w through the stochastic gradient update
            w = w - gamma * gradient
            
            # Calculate loss
            loss = compute_loss_mse(y, tx, w)
   
    return (w, loss)

def least_squares(y, tx) : 
    """Calculate the least squares solution.
        returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    
    """
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    e = y - tx@w
    mse = 1/(2*y.shape[0]) * e.T@e
    return(w, float(mse))

def ridge_regression(y, tx, lambda_) :
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    
    """
    w = np.linalg.solve(
        tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]), 
        tx.T @ y
    )
    loss = compute_loss_mse(y, tx, w) 
    return (w, loss)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """

    sigma = 1.0 / (1.0 + np.exp(-t))
    return sigma

def calculate_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx @ w)
    loss = -(1/ y.shape[0]) * np.sum(
        y * np.log(pred) + (1 - y) * np.log(1 - pred)
    )
    return float(loss)

def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    grad = 1/y.shape[0] * tx.T@(sigmoid(tx@w)-y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    grad = calculate_gradient_logistic(y, tx, w)
    loss = calculate_loss_logistic(y, tx, w)
    w = w - gamma * grad

    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    '''
    Logistic regression using gradient descent

    Args : 
        y : shape(N, 1)
        tx : shape(N, D)
        initial_w : shape(D, 1)
        max_inters : int
        gamma : float

    Returns : 
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    '''
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    return (w, loss)

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss = calculate_loss_logistic(y, tx, w)
    gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    w = w - gamma*gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma) :
    '''
    Regularized logistic regression using gradient descent

    Args : 
        y : shape(N, 1)
        tx : shape(N, D)
        initial_w : shape(D, 1)
        max_inters : int
        gamma : float

    Returns : 
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    '''
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    return (w, loss)
