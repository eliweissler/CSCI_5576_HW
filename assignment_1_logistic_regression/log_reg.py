from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def grad_LCE_log_reg(X: np.array, Y: np.array, pred: np.array):
    """
    Returns the gradient of the Cross Entropy loss function

    Returns [grad with respect to w, grad with respect to b]

    Args:
        X (np.array): Data matrix (n data points rows, m dimensions columns)
        Y (np.array): Actual labels for dataset (n rows, 1 column)
        pred (np.array): Predicted labels for the data (n rows, 1 column)
    Returns:
        gradient with respect to w (np.array 1 row, m columns),
        deriv with respect to b (float)
    """

    diff = pred - Y
    grad_w = (1/X.shape[0])*diff.T@X
    deriv_b = np.mean(diff)

    return grad_w, deriv_b


def loss_LCE(Y: np.array, pred: np.array):
    """
    Returns the cross entropy loss for the specified
    data labels and predictions

    Args:
        Y (np.array): Actual labels for the dataset (n rows, 1 column)
        pred (np.array): Predicted labels for the data (n rows, 1 column)

    Returns:
        loss function value (float)
    """
    total_loss = 0
    for i in range(len(Y)):
        # 1 Label
        if Y[i]:
            total_loss += -np.log2(pred[i])
        # 0 label
        else:
            total_loss += -np.log2(1-pred[i])
    return total_loss/Y.shape[0]


def sigmoid(x):
    """Simple sigmoid function S(x) = 1/(1 + e^-x)"""
    return 1/(1 + np.exp(-x))


def pred_sigmoid(X: np.array, W: np.array, b: float):
    """
    Returns the predicted labels for the dataset using 
    the sigmoid function

    Args:
        X (np.array): Data matrix (n data points rows, m dimensions columns)
        W (np.array): Weight matrix (1 row, m columns)
        b (float): Additive bias

    Returns:
        predictions (np.array n rows, 1 column)
    """
    z = X@W.T + b
    return sigmoid(z)


def grad_descent(X: np.array, Y: np.array,
                 loss_fn: Callable, grad_fn: Callable, pred_fn: Callable,
                 epochs: int, lr: float):
    """
    Performs gradient descent on the data for a specified loss function
    and gradient function.

    Args:
        X (np.array): data matrix
        Y (np.array): data labels
        loss_fn (function): function that takes in (Y, pred) -> float
                            that is the error.
        grad_fn (function): function that takes in (X, Y, pred) -> [np.array,
                                                                    float]
                            that is [d/dW Loss, d/db Loss].
        pred_fn (function): function that takes in (X, W, b) -> np.array
                            that is the predicted labels.
        epochs (int): number of epochs
        lr (float): learning rate (W_{i+1} = W_{i} + lr*grad)

    Returns:
        W, b, pred, loss
    """
    loss = np.zeros(epochs)

    # Initialize weights as all ones
    # with zero bias
    W = np.ones(X.shape[1])
    b = 0

    for i in range(epochs):

        # Predictions for this epoch
        pred = pred_fn(X, W, b)

        # Loss function value with the current preds
        loss[i] = loss_fn(Y, pred)

        # Update weights
        if i < epochs - 1:
            grad = grad_fn(X, Y, pred)
            W += -lr*grad[0]
            b += -lr*grad[1]

    return W, b, pred, loss


def normalize_data(data: pd.DataFrame, label_col: str = "LABEL"):
    """
    Creates a data matrix X and label matrix Y out of 
    the provided dataframe.

    Maps the label column to integers between 0-num_unique_labels.

    Normalizes the data columns to be between 0-1, 
    with 0 as the minimum value found, and 1 as
    the maximum value found.

    ignores the column if the min/max are the same

    Args:
        data (pd.DataFrame): DataFrame to process
        label_col (str, optional): Column with the data label in it.
                                   Defaults to "LABEL".

    Returns:
        X, Y (np.arrays), label_mapping (dictionary)
    """
    # Get labels
    labels = data[label_col].values
    Y = np.zeros(labels.size, dtype=int)
    label_map = {}
    count = 0
    for i, label in enumerate(labels):
        # Add mapping to dictionary
        if label not in label_map:
            label_map[label] = count
            count += 1
        # Assign value in Y
        Y[i] = label_map[label]
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
                 err_plot: str = "", conf_mat: str = ""):
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
                        put no filename to display the figure.
        conf_mat (str): file to save a plot of the confusion matrix to.
                        put no filename to display the figure.
    """
    # Read in and normalize data
    data = pd.read_csv(data_file)
    X, Y = normalize_data(data, label_col=label_col)

    # Do the gradient descent
    W, b, pred, loss = grad_descent(X, Y, loss_LCE, grad_LCE_log_reg,
                                    pred_sigmoid, epochs=epochs, lr=lr)

    # Make plots

    # Error
    f, ax = plt.subplots()
    ax.plot(np.arange(epochs), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_{CE}$")
    ax.set_title(f"Cross Entropy Loss (LR = {lr})")
    if err_plot != "":
        plt.savefig(err_plot)
        plt.close(f)
    else:
        plt.show()

    # Confusion Mat
    cm = confusion_matrix(Y, (pred > 0.5).astype(int))
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Good Day", "Bad Day"])
    ax.yaxis.set_ticklabels(["Good Day", "Bad Day"])
    if err_plot != "":
        plt.savefig(conf_mat)
        plt.close(f)
    else:
        plt.show()

    return W, b, pred, loss
