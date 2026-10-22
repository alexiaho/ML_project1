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
    gradient = -1 / y.shape[0] * tx.T @ e
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w.copy()

    # Compute the gradient and update the weights for max_iters iterations
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    # Compute loss after final weight update (works also if max_iters == 0)
    loss = compute_loss_mse(y, tx, w)  # Loss computed after final weight update

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


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w.copy()

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):

            # Compute a stochastic gradient and loss
            gradient = compute_gradient(
                y_batch, tx_batch, w
            )  # gradient computed on small batch

            # Update w through the stochastic gradient update
            w = w - gamma * gradient

            # Calculate loss
            loss = compute_loss_mse(y, tx, w)

    return (w, loss)


def least_squares(y, tx):
    """Calculate the least squares solution.
        returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    mse = 1 / (2 * y.shape[0]) * e.T @ e
    return (w, mse)  # Loss should ba a NumPy scalar, not a float


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)

    """
    w = np.linalg.solve(
        tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]), tx.T @ y
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
    loss = -(1 / y.shape[0]) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss  # Return a NumPy scalar, not a float


def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    grad = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]
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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    Args :
        y : shape(N, 1)
        tx : shape(N, D)
        initial_w : shape(D, 1)
        max_inters : int
        gamma : float

    Returns :
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    """
    w = initial_w.copy()

    # Perform gradient descent for max_iters iterations
    for _ in range(max_iters):
        _, w = learning_by_gradient_descent(y, tx, w, gamma)

    # Compute loss after final weight update (works also if max_iters == 0)
    loss = calculate_loss_logistic(y, tx, w)

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
    w = w - gamma * gradient
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    Args :
        y : shape (N, 1)
        tx : shape (N, D)
        initial_w : shape (D, 1)
        max_iters : int
        gamma : float

    Returns :
        (w, loss): last weight vector of the method, and the corresponding loss value (cost function)
    """
    w = initial_w.copy()

    # Perform penalized gradient descent for max_iters iterations
    for _ in range(max_iters):
        _, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

    # Compute loss after final weight update (works also if max_iters == 0)
    loss = calculate_loss_logistic(y, tx, w)

    return (w, loss)

def weighted_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Class-weighted regularized logistic regression (gradient descent).

    Works with 1-D arrays (as required by the project) and also with (N,1)/(D,1)
    column vectors: everything is handled as 1-D internally, and w is returned
    with the same shape as initial_w.
    """
    y = np.asarray(y).ravel()                      # (N,)
    w = np.asarray(initial_w, dtype=float).ravel()  # (D,)

    # Calculate weights from the training labels
    negative_rate = np.mean(y == 0)
    positive_rate = np.mean(y == 1)

    weight_negative = 1.0 / (2 * negative_rate)
    weight_positive = 1.0 / (2 * positive_rate)

    # Give each training example its class weight
    sample_weights = np.where(y == 1, weight_positive, weight_negative)  # (N,)

    for _ in range(max_iters):
        probabilities = sigmoid(tx @ w)

        # Weighted logistic gradient
        gradient = tx.T @ (sample_weights * (probabilities - y)) / len(y)

        # Regularization, matching reg_logistic_regression
        gradient += 2 * lambda_ * w

        w = w - gamma * gradient

    # Calculate weighted logistic loss safely
    probabilities = np.clip(sigmoid(tx @ w), 1e-15, 1 - 1e-15)

    loss = -np.mean(
        sample_weights * (
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )
    ) + lambda_ * np.sum(w**2)

    return w.reshape(np.shape(initial_w)), loss


def one_hot_encode_categorical(x_train, x_test, min_categories=2, max_categories=6):
    """
    Safely auto-detects true low-cardinality categorical columns (e.g. scales 1-5, binary 0-1).
    """
    x_tr = x_train.copy()
    x_te = x_test.copy()
    
    n_samples, n_features = x_tr.shape
    categorical_indices = []
    
    sample_size = min(5000, n_samples)
    sample_data = x_tr[:sample_size, :]
    
    for i in range(n_features):
        col_vals = sample_data[:, i]
        valid_vals = col_vals[~np.isnan(col_vals)]
        if len(valid_vals) == 0:
            continue
        unique_vals = np.unique(valid_vals)
        
        # Stricter check: Must be integer-like AND have between 2 and 6 unique values
        if min_categories <= len(unique_vals) <= max_categories and np.all(unique_vals == unique_vals.astype(int)):
            categorical_indices.append(i)
            
    print(f"Smart-detected {len(categorical_indices)} true categorical columns out of {n_features}.")
    
    tr_parts = []
    te_parts = []
    
    for i in range(n_features):
        if i in categorical_indices:
            col_tr = x_tr[:, i]
            col_te = x_te[:, i]
            
            col_tr_filled = np.nan_to_num(col_tr, nan=-999)
            col_te_filled = np.nan_to_num(col_te, nan=-999)
            
            categories = np.unique(col_tr_filled)
            
            oh_tr = (col_tr_filled[:, None] == categories).astype(float)
            oh_te = (col_te_filled[:, None] == categories).astype(float)
            
            tr_parts.append(oh_tr)
            te_parts.append(oh_te)
        else:
            tr_parts.append(x_tr[:, i:i+1])
            te_parts.append(x_te[:, i:i+1])
            
    return np.hstack(tr_parts), np.hstack(te_parts)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_pred = (probabilities >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / len(y_true)

    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }


def find_best_threshold(y_true, probabilities, decision_thresholds):
    results = []

    for threshold in decision_thresholds:
        metrics = calculate_metrics(
            y_true, probabilities, threshold
        )
        results.append({
            "Threshold": round(float(threshold), 2),
            **metrics
        })

    results.sort(key=lambda row: row["F1"], reverse=True)
    return results


def make_stratified_folds(y, n_folds=5, seed=42):
    """Create stratified folds for binary labels encoded as 0 and 1."""
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)

    class_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        class_indices.append(np.array_split(indices, n_folds))

    folds = []
    for k in range(n_folds):
        fold_indices = np.concatenate([
            class_indices[0][k],
            class_indices[1][k]
        ])
        rng.shuffle(fold_indices)
        folds.append(fold_indices)

    return folds


def sigmoid_safe(z):
    """Numerically stable sigmoid."""
    z = np.asarray(z)
    result = np.empty_like(z, dtype=float)

    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)

    return result


def train_logistic_sgd(
    y, X, lambda_=0.0005, learning_rate=0.05,
    epochs=12, batch_size=2048, weighted=False, seed=42
):
    """
    Train binary logistic regression using shuffled mini-batch SGD.

    y must contain 0/1 labels.
    X is expected to include the bias column (first column) from preprocess_data.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))

    if weighted:
        negative_rate = np.mean(y.ravel() == 0)
        positive_rate = np.mean(y.ravel() == 1)

        class_weights = np.where(
            y == 1,
            1.0 / (2.0 * positive_rate),
            1.0 / (2.0 * negative_rate)
        )
    else:
        class_weights = np.ones_like(y)

    # Avoid regularizing the bias column (first column, added in preprocess_data).
    regularization_mask = np.ones((n_features, 1))
    regularization_mask[0, 0] = 0.0

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        shuffled_indices = rng.permutation(n_samples)
        current_lr = learning_rate / (1.0 + 0.1 * epoch)

        for start in range(0, n_samples, batch_size):
            batch_idx = shuffled_indices[start:start + batch_size]

            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            weights_batch = class_weights[batch_idx]

            probabilities = sigmoid_safe(X_batch @ w)

            gradient = (
                X_batch.T @ (
                    weights_batch * (probabilities - y_batch)
                ) / len(batch_idx)
            )

            gradient += (
                2.0 * lambda_ * regularization_mask * w
            )

            w -= current_lr * gradient

    return w