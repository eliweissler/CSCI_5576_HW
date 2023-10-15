from typing import Callable

from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def loss_MSE(Y: np.array, pred: np.array, deriv: bool = False):
    """
    Returns the Mean Squared Error Loss (NOTE: Calculates entry for each
    element, you need to take mean to get the mean)

    Args:
        Y (np.array): Actual labels for the dataset (n rows, 1 column)
        pred (np.array): Predicted labels for the data (n rows, 1 column)
        deriv (bool): Whether to return the derivative with respect to predictions.

    Returns:
        loss function value (float)
    """
    if deriv:
        return pred - Y
    else:
        return (1/2)*((pred - Y)**2)


def sigmoid(x: float, deriv: bool = False):
    """Simple sigmoid function S(x) = 1/(1 + e^-x)"""
    if deriv:
        return sigmoid(x, deriv=False)*(1-sigmoid(x, deriv=False))
    else:
        return 1/(1 + np.exp(-x))

class NeuralNetwork:

    def __init__(self, input_size: int, output_size: int, 
                 hidden_layer_sizes: list[int], activ_funcs: list[Callable],
                 loss_func: Callable, random_initialize: bool = False):
        """
        Initializes a Neural Network with Specified shapes/activation functions

        Args:
            input_size (int): input size, i.e. number of dimensions of data
            output_size (int): output size, i.e. dimension of output
            hidden_layer_sizes (list[int]): number of neurons in each hidden layer
            activ_funcs (list[Callable]): activation functions for each layer.
                                          Should provide number of hidden layers
                                          plus one functions. Function needs to
                                          take inputs of (x, deriv).
                                          input -> z1 (activ) h1 -> z2 (activ) output
            loss_func (Callable): loss function. Must have arguments of 
                                  (Y, pred, deriv).
        """

        # Record activation functions
        if len(activ_funcs) != len(hidden_layer_sizes) + 1:
            raise ValueError("Not the right number of activation functions")
        else:
            self.activ_fns = activ_funcs

        # Record loss function
        self.loss_fn = loss_func

        # Initialize Neuron Values for non-inputs
        self.layers = [np.zeros((1, s)) for s in hidden_layer_sizes] + \
                      [np.zeros((1, output_size))]
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.input_size = input_size
        self.output_size = output_size

        # Initialize Weights and biases
        all_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = []
        self.biases = []
        for i in range(len(all_sizes) - 1):
            if random_initialize:
                self.weights.append(np.random.random((all_sizes[i], all_sizes[i+1])))
                self.biases.append(np.random.random((1, all_sizes[i+1])))
            else:
                self.weights.append((i+1)*np.ones((all_sizes[i], all_sizes[i+1])))
                self.biases.append(np.zeros((1, all_sizes[i+1])))
    
    def calc_jacobian(self):
        """
        Calculates the jacobian
        (size m output layer, n-1 hidden layers)
        [[ ds1/dz1, ds1/dz2 ... ds1/dzn ],
         [ ds2/dz1, ds2/dz2 ... ds2/dzn ],
         [ ...      ...          ...    ],
         [ dsm/dz1, dsm/dz1 ... dsm/dzn ]]

        Returns:
            np.array: Jacobian matrix
        """
        
        J = np.zeros((self.output_size, self.n_hidden_layers + 1))

        # Start from the right of the jacobian, i.e. z value for the 
        # output layer
        for zn in range(J.shape[1])[::-1]:
            # For output layer
            if zn == J.shape[1] - 1:
                J[:, zn] = np.mean(self.activ_fns[zn](self.layers[zn], deriv=True))
            # Hiden layer
            else:
                J[:, zn] = np.mean((self.activ_fns[zn](self.layers[zn], deriv=True)@self.weights[zn+1])*J[:, zn+1])

        return J
    
    def train(self, X: np.array, Y: np.array, X_test: np.array, Y_test: np.array, 
              epochs: int, lr: float, batch_size: int = None, check_progress: int = 1000):
        """
        Trains the network via backpropagation and gradient descent

        Args:
            X (np.array): data to train on
            Y (np.array): data labels for training data
            X_test (np.array): data to report loss with respect to
            Y_test (np.array): data labels for test data
            epochs (int): number of epochs
            lr (float): learning rate (W_{i+1} = W_{i} - lr*grad)
            batch_size (int, optional): randomly selects the given
                                        number of elements each epoch.
            check_progres (int): Print out progress after this many epochs
        
        Returns:
            loss
        """

        loss = np.zeros(epochs)

        for n in range(epochs):
            # Select a data subset
            if batch_size is None:
                subset_X = X
                subset_Y = Y
            else:
                indx = np.random.choice(X.shape[0], batch_size, replace=False)
                subset_X = X[indx]
                subset_Y = Y[indx]

            # Feed forward through network
            pred = self.feed_forward(subset_X)

            # Backpropagation to update gradients
            J = self.calc_jacobian()
            self.back_propagate(subset_X, subset_Y, pred, J, lr)

            
            # Calculate loss
            pred = self.feed_forward(X_test)
            # breakpoint()
            loss[n] = np.mean(self.loss_fn(Y_test, pred, deriv=False))

            # Print Progress
            if n % check_progress == 0:
                print(f"Epoch {n} (out of {epochs}) -- Loss: {np.round(loss[n], 4)}")


        return loss
    
    def feed_forward(self, X: np.array):
        """
        Feeds forward an input through the network. 
        Returns the output layer.

        Args:
            X (np.array): data matrix

        Returns:
            np.array: output layer post activation function
        """

        # Go through each layer and apply weights
        # Plus activation function
        prev_layer = X
        for i in range(self.n_hidden_layers + 1):
            self.layers[i] = prev_layer@self.weights[i] + self.biases[i]
            prev_layer = self.activ_fns[i](self.layers[i])

        return prev_layer
    
    def back_propagate(self, X: np.array, Y: np.array, pred: np.array,
                       J: np.array, lr: float):
        """
        Updates weights and biases using backpropagation

        Args:
            X (np.array): data matrix
            Y (np.array): Actual labels for the dataset (n rows, 1 column)
            pred (np.array): Predicted labels for the data (n rows, 1 column)
            J (np.array): Jacobian matrix
            lr (float): learning rate
        """

        dL_dyhat = self.loss_fn(Y, pred, deriv=True)

        # Go through and update weights and biases for
        # each layer
        for n in range(self.n_hidden_layers + 1):
            root = dL_dyhat*J[:, n]
            if n > 0:
                self.weights[n] += -lr*np.mean(root*self.activ_fns[n-1](self.layers[n-1]))
            else:
                self.weights[n] += -lr*np.mean(root*X)   
            self.biases[n] += -lr*np.mean(root)
        
        return

    



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
    Y = np.zeros((labels.size, 1), dtype=int)
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


def plot_confusion_matrix(Y: np.array, pred: np.array, savename=""):
    """
    Convenience function for generating a confusion Matrix

    Args:
        Y (np.array): Actual labels for the dataset (n rows, 1 column)
        pred (np.array): Predicted labels for the data (n rows, 1 column)
        savename (str, optional): File to save plot to. If none is given shows figure.
                                    Defaults to "".

    Returns:
        confusion matrix
    """
    # Figure out predicted class -- infer from Y and pred the number of classes
    cm = confusion_matrix(Y, (pred > 0.5).astype(int))
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    if savename != "":
        plt.savefig(savename)
        plt.close(f)
    else:
        plt.show()

    return cm


def plot_loss(loss: np.array, lr: float, savename="", yscale="linear"):
    """
    Convenience function for plotting loss over epochs.
    Assumes loss provided is per epoch.

    Args:
        loss (np.array): loss over epochs
        lr (float): learning rate to display in plot
        savename (str, optional): File to save plot to. If none is given shows figure.
                                    Defaults to "".
        yscale (str, optional): matplotlib set_yscale argument. Defaults to linear.
    """
    
    # Error
    f, ax = plt.subplots()
    ax.plot(np.arange(len(loss)), loss)
    ax.set_yscale(yscale)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_{MSE}$")
    ax.set_title(f"Loss (LR = {lr})")
    if savename != "":
        plt.savefig(err_plot)
        plt.close(f)
    else:
        plt.show()



if __name__ == "__main__":


    ##################################### PART 1

    # Initialize network and load data
    data = pd.read_csv("A2_Data_EliWeissler.csv")
    # data = pd.read_csv("A1_Data_EliWeissler.csv")
    X, Y = normalize_data(data)

    input_size = 3
    hidden_layers = [4]
    output_size = 1
    activation_fns = [sigmoid, sigmoid]
    loss_fn = loss_MSE
    random_initialize = True
    network = NeuralNetwork(input_size, output_size, hidden_layers,
                            activation_fns, loss_fn, random_initialize=random_initialize)
    
    # Train network
    epochs = 10000
    lr = 0.1
    batch_size = 1
    loss = network.train(X, Y, X, Y, epochs=epochs, lr=lr, batch_size=batch_size)

    # Predict
    pred = network.feed_forward(X)

    # Plot Titles
    err_plot = "err.png"
    conf_mat = "conf.png"

    plot_confusion_matrix(Y, pred, conf_mat)
    plot_loss(loss, err_plot)



    
    
    ##################################### PART 2