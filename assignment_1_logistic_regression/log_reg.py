import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def grad_LCE_log_reg():
    return

def loss_LCE():
    return

def pred_sigmoid():
    return


def grad_descent(X: np.array, Y: np.array,
                 loss_fn: function, grad_fn: function, pred_fn: function,
                 epochs: int, lr: float):
    """
    Performs gradient descent on the data for a specified loss function
    and gradient function.

    Args:
        X (np.array): data matrix
        Y (np.array): data labels
        loss_fn (function): function that takes in (X, Y, pred) -> float
                            that is the error.
        grad_fn (function): function that takes in (X, Y, pred) -> [np.array, np.array]
                            that is [d/dW Loss, d/db Loss].
        pred_fn (function): function that takes in (X, W, b) -> np.array
                            that is the predicted labels.
        epochs (int): number of epochs
        lr (float): learning rate (W_{i+1} = W_{i} + lr*grad)

    Returns:
        (W, b), loss
    """
    loss = np.zeros(epochs)

    # Initialize weights as all ones
    # with zero bias
    W = np.ones(size=X.shape).T
    b = 0

    for i in range(epochs):

        # Predictions for this epoch
        pred = pred_fn(X, W, b)

        # Loss function value with the current preds
        loss[i] = loss_fn(X, Y, pred)

        # Update weights
        grad = grad_fn(X, Y, pred)
        W += lr*grad[0]
        b += lr*grad[1]

    return


def normalize_data(data: pd.DataFrame, label_col: str = "LABEL"):
    """
    Creates a data matrix X and label matrix Y out of 
    the provided dataframe.

    Normalizes the data columns to be between 0-1, 
    with 0 as the minimum value found, and 1 as
    the maximum value found.

    ignores the column if the min/max are the same

    Args:
        data (pd.DataFrame): DataFrame to process
        label_col (str, optional): Column with the data label in it.
                                   Defaults to "LABEL".

    Returns:
        X, Y (np.arrays)
    """
    # Get labels
    Y = data[label_col].values

    # Normalize columns
    X = np.zeros((data.shape[0], data.shape[1] - 1))
    cols_to_keep = np.zeros(data.shape[1]-1, dtype=bool)
    for j, col in enumerate(data.columns):
        if col != label_col:
            # Check min/max values
            dmin = data[col].min()
            dmax = data[col].max()
            # Save data if it's not uniform
            if dmax > dmin:
                X[:, j] = (data[col] - dmin)/(dmax - dmin)
                cols_to_keep[j] = True

    # Only keep non-uniform columns
    X = X[:, cols_to_keep]

    return X, Y


def logistic_reg(data_file: str, epochs: int, lr: float,
                 label_col: str = "LABEL",
                 err_plot: str = "error.png", conf_mat: str = "conf_mat.png"):
    """
    Performs a logistic regression using gradient descent

    Args:
        data_file (str): csv file containing the labeled dataset.
                         Assumes all columns that aren't the label
                         column are for prediction.
        epochs (int): number of epochs to run.
        lr (float): learning rate for gradient descent.
        label_col (str): label column within the data.
        err_plot (str): file to save a plot of the error function vs. epoch.
        conf_mat (str): file to save a plot of the confusion matrix to.
    """
    # Read in and normalize data
    data = pd.read_csv(data_file)
    X, Y = normalize_data(data, label_col=label_col)

    # Do the gradient descent
    (W, b), loss = grad_descent(X, Y, loss_LCE, grad_LCE_log_reg, pred_sigmoid,
                                epochs=epochs, lr=lr)

    # Make plots

    # Error
    f = plt.figure()
    plt.savefig(err_plot)
    plt.close(f)

    # Confusion Mat
    f = plt.figure()
    plt.savefig(conf_mat)
    plt.close(f)
