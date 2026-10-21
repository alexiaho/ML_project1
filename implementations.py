import numpy as np

#used in mean_squared_error_gd and mean_squared_error_sgd
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

#used in mean_squared_error_gd and mean_squared_error_sgd
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

#used in mean_squared_error_sgd
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

#used in calculate_loss_logistic and calculate_gradient_logistic
def sigmoid(z):
    """apply sigmoid function on t. Can handle large z without overflow

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """

    z = np.asarray(z)
    result = np.empty_like(z, dtype=float)
    
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    
    return result


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
    grad = 1 / y.shape[0] * tx.T @ (sigmoid(tx @ w) - y)
    return grad


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
        grad = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        

    # Compute loss after final weight update (works also if max_iters == 0)
    loss = calculate_loss_logistic(y, tx, w)

    return (w, loss)


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
        gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    # Compute loss after final weight update (works also if max_iters == 0)
    loss = calculate_loss_logistic(y, tx, w)

    return (w, loss)

def preprocess_data(x_train, x_test, nan_threshold=0.90):
    """
    Perform preprocessing on training and testing datasets. Includes filtering Nan values, replacing missing values with column medians (imputation), and scaling.
    
    Parameters:
    - x_train: array of raw training features 
    - x_test: array of raw testing features 
    - nan_threshold: column NaN percentage limit to keep (e.g. 0.90 = keep if <= 90% NaN)
    
    Returns:
    - x_train_final: filtered, imputed and scaled training set 
    - x_test_final: filtered, imputed and scaled test set 
    """
    print("--- Starting Preprocessing ---")
    
    # 1. Remove columns with high NaN ratios (> nan_threshold)
    nan_ratios_columns = np.isnan(x_train).mean(axis=0)
    columns_to_keep = nan_ratios_columns <= nan_threshold
    
    x_tr_filtered = x_train[:, columns_to_keep].copy()
    x_te_filtered = x_test[:, columns_to_keep].copy()

    nb_columns_removed = np.sum(~columns_to_keep) # The ~ operator performs a logical NOT on Boolean values --> False becomes True and vice vera
    percentage_nan_threshold = nan_threshold * 100

    print(f'Number of removed columns with >{percentage_nan_threshold:.0f}% NaNs: {nb_columns_removed}')
    print(f'Remaining features: {x_tr_filtered.shape[1]}')
    
    # 2. Replace NaN values by column median values
    train_medians = np.nanmedian(x_tr_filtered, axis=0) # computes median along axis=0, ignoring existing NaN values 

    # Scans the array of computed medians and converts any NaN values to 0.0
    train_medians = np.nan_to_num(train_medians, nan=0.0) 
    
    for col_idx in range(x_tr_filtered.shape[1]):
        # Replaces missing values in training set with corresponding column median
        train_nan = np.isnan(x_tr_filtered[:, col_idx])
        x_tr_filtered[train_nan, col_idx] = train_medians[col_idx]
        
        # Replaces missing values in testing set with corresponding column median from training set 
        test_nan = np.isnan(x_te_filtered[:, col_idx])
        x_te_filtered[test_nan, col_idx] = train_medians[col_idx]
        
    print("Missing values replaced using medians from the training set.")
    
    # 3. Standardization (Z-score normalization)
    mean = np.mean(x_tr_filtered, axis=0)
    std = np.std(x_tr_filtered, axis=0)
    
    # Special case: division by zero --> std replaced by 1.0 for constant columns
    std[std == 0] = 1.0
    
    x_tr_normalized = (x_tr_filtered - mean) / std
    x_te_normalized = (x_te_filtered - mean) / std
    print("Features standardized (zero mean, unit variance).")
    
    # 4. Add a column of 1s at the beginning for the bias weight (w_0)
    bias_train = np.ones((x_tr_normalized.shape[0], 1))
    bias_test = np.ones((x_te_normalized.shape[0], 1))
    
    x_tr_final = np.hstack((bias_train, x_tr_normalized))
    x_te_final = np.hstack((bias_test, x_te_normalized))
    
    print(f"Final processed training shape: {x_tr_final.shape}")
    print(f"Final processed testing shape: {x_te_final.shape}")
    print("--- Preprocessing Complete ---\n")
    
    return x_tr_final, x_te_final


def correlation_filtering(x_train, x_test, threshold=0.95):
    """
    Filter out features from the training and testing datasets that are highly correlated with each other.
    
    Parameters:
    - x_train: array of training features 
    - x_test: array of testing features 
    - threshold: correlation threshold above which features will be removed
    
    Returns:
    - x_train_filtered: filtered training set 
    - x_test_filtered: filtered test set 
    """
    print("--- Starting Correlation Filtering ---")
    
    # Compute correlation matrix for training data
    correlation = np.corrcoef(x_train, rowvar=False)
    
    to_remove = set()  # Indices of features to remove
    n_features = correlation.shape[0]

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(correlation[i, j]) > threshold:
                to_remove.add(j)

    selected_indices = [i for i in range(n_features) if i not in to_remove]

    x_train_filtered = x_train[:, selected_indices]
    x_test_filtered = x_test[:, selected_indices]
    
    print(f"Number of removed features due to high correlation: {len(to_remove)}")
    print(f"Remaining features after filtering: {x_train_filtered.shape[1]}")
    print("--- Correlation Filtering Complete ---\n")
    
    return x_train_filtered, x_test_filtered

def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma
):
    """Class-weighted regularized logistic regression.
    
    Args : 

    y : (N,)
    tx : (N, D)
    lambda_ : scalar, regularization factor
    initial_w : (D,) : initial weight vector
    max_iters : int, number of iterations of the gradient descent
    gamma : scalar, learning rate

    Returns : 

    w : optimal weight vector
    loss : loss associated with these weights
    """
    w = initial_w.copy()

    # Calculate weights from the training labels
    negative_rate = np.mean(y.ravel() == 0)
    positive_rate = np.mean(y.ravel() == 1)

    weight_negative = 1.0 / (2 * negative_rate)
    weight_positive = 1.0 / (2 * positive_rate)

    # Give each training example its class weight
    sample_weights = np.where(
        y == 1, weight_positive, weight_negative
    ).reshape(-1, 1)

    w, _ = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    

    # Calculate weighted logistic loss safely
    probabilities = np.clip(sigmoid(tx @ w), 1e-15, 1 - 1e-15)

    loss = -np.mean(
        sample_weights * (
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )
    ) + lambda_ * np.sum(w**2)

    return w, loss


def make_stratified_folds(y, n_folds=5, seed=42):
    """Create stratified folds for binary labels encoded as 0 and 1.
    
    Args:
    
    y :(N,)
    n_folds : int, number of folds
    seed : int, to ensure reproducibility
    
    Returns :
    
    folds : array of indices"""

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

#pas sure du batch size on nous dit 1 dans l'énoncé mais cette fonction est trop bizarre
#anyways faudra essayer de la réécrire
def train_logistic_sgd(
    y, X, lambda_=0.0005, learning_rate=0.05,
    epochs=12, batch_size=2048, weighted=False, seed=42
):
    """
    Train binary logistic regression using shuffled mini-batch SGD.

    Args : 
    y : (N,), must contain 0/1 labels.
    X  : (N, D+1), is expected to include the bias column from preprocess_data.
    lambda_ : scalar, regularization factor
    learning_rate : scalar
    epochs : int, number of passes through training data
    batch_size : int
    weighted : bool, weighted version or not
    seed : int, for reproducibility

    Returns : 
    w : optimal weights vector
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

    # Avoid regularizing the final bias column.
    regularization_mask = np.ones((n_features, 1))
    regularization_mask[-1, 0] = 0.0

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        shuffled_indices = rng.permutation(n_samples)
        current_lr = learning_rate / (1.0 + 0.1 * epoch)

        for start in range(0, n_samples, batch_size):
            batch_idx = shuffled_indices[start:start + batch_size]

            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            weights_batch = class_weights[batch_idx]

            probabilities = sigmoid(X_batch @ w)

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


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """ Calculates the F1 score, the precision, the recall and the accuracy of the model

    Args :
    y_true : (N,), actual values of y for the test set
    probabilities : (N ,), probabilities resulting of the logistic regression
    threshold : scalar

    Returns :
    F1 : scalar
    Precision : scalar
    Recall : scalar
    Accuracy : scalar
    
    """
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


def find_best_threshold(y_true, probabilities):
    """Compares different threshold and finds the best one
    
    Args:
    y_true : (N,), actual values of y for the test set
    probabilities : (N ,), probabilities resulting of the logistic regression

    Returns : 

    results : the different thresholds and their corresponding metrics, sorted by highest F1 score
    """
    thresholds = np.arange(0.05, 0.801, 0.01)
    results = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_true, probabilities, threshold
        )
        results.append({
            "Threshold": round(float(threshold), 2),
            **metrics
        })

    results.sort(key=lambda row: row["F1"], reverse=True)
    return results
