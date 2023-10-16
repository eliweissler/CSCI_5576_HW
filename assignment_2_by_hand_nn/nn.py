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


def loss_CCE(Y: np.array, pred: np.array, deriv: bool = False):
    """
    Returns the Catigorical Cross Entropy Loss (NOTE: Calculates entry
    for each element, you need to take mean to get the mean)

    Args:
        Y (np.array): Actual labels for the dataset (n rows, m columns)
        pred (np.array): Predicted labels for the data (n rows, m columns)
        deriv (bool): Whether to return the derivative with respect to predictions.

    Returns:
        loss function value (float)
    """
    # Epsilon value to avoid divide by 0 or log of 0
    eps = 1e-10
    pred_clipped = pred.copy()
    pred_clipped[pred_clipped < eps] = eps
    if deriv:
        return -Y/pred_clipped
    else:
        return -Y*np.log2(pred_clipped)


def sigmoid(x: float, deriv: bool = False):
    """
    Simple sigmoid function S(x) = 1/(1 + e^-x)
    Args:
        x (np.array): input data, each row is an entry
        deriv (bool, optional): Return derivative w.r.t. input vector.
                                Defaults to False.

    Returns:
        2D array where each row is S(x_i) or d/dx_i S(x_i)or
        3D array where each entry is a Jacobian dsigmoid(x)_i/dx_j
        """
    if deriv:
        vals = sigmoid(x, deriv=False)*(1-sigmoid(x, deriv=False))
        return np.array([np.diag(val) for val in vals])
    else:
        return 1/(1 + np.exp(-x))


def softmax(x: np.array, deriv: bool = False):
    """
    Softmax function s(x)_i = exp(x_i)/(sum_j exp(x_j))

    Args:
        x (np.array): input data, each row is an entry
        deriv (bool, optional): Return derivative w.r.t. input vector.
                                Defaults to False.

    Returns:
        2D Array where each row is s(x)_i or 
        3D Array where each entry is a Jacobian ds(x)_i/dx_j
    """
    # Avoid overflows in exp
    upper_clip = 50
    lower_clip = -50
    if deriv:
        dim = x.size
        if len(x.shape) == 2:
            dim = x.shape[1]
        sx = softmax(x, deriv=False)
        J = np.array([np.diag(sxi) - np.outer(sxi, sxi) for sxi in sx])
        return J
    else:
        sx = np.array([np.exp(np.clip(xi, a_min=lower_clip, a_max=upper_clip)) for xi in x])
        for i in range(sx.shape[0]):
            sx[i] /= sx[i].sum()
        return sx
    

def ReLU(x: np.array, deriv: bool = False):
    """
    Simple ReLU Function ReLU(x) = x if x > 0 else 0 

    Args:
        x (np.array): input data, each row is an entry
        deriv (bool, optional): Return derivative w.r.t. input vector.
                                Defaults to False.

    Returns:
       2D array where each row is ReLU(x_i) or
       3D array where each entry is a Jacobian dReLU(x)_i/dx_j
    """
    if deriv:
        vals = np.heaviside(x, 0)
        return np.array([np.diag(val) for val in vals])
    else:
        return x*np.heaviside(x, 0)


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

        # Initialize Blank Jacobians and gradients with respect to loss
        self.J = []
        self.dLdz = []
        for n in range(self.n_hidden_layers + 1):
            if n == self.n_hidden_layers:
                self.J.append(np.zeros((output_size, output_size)))
                self.dLdz.append(np.zeros((1, output_size)))
            else:
                self.J.append(np.zeros((output_size, hidden_layer_sizes[n])))
                self.dLdz.append(np.zeros((1, hidden_layer_sizes[n])))
    
    def calc_jacobian(self):
        """
        Calculates the jacobian for the output layer
        with respect to every hidden layer
        (size m output layer, n-1 hidden layers)

        J is a list where each entry is a jacobian matrix
        J = [... doi/dznj ...]

        [[ do1/dz1, do1/dz2 ... do1/dzn ],
         [ do2/dz1, do2/dz2 ... do2/dzn ],
         [ ...      ...          ...    ],
         [ dom/dz1, dom/dz1 ... dom/dzn ]]

         using the values from the most recent call of feed_forward

        Returns:
            np.array: Jacobian matrix
        """
        
        # Start from the right of the self.Jacobian, i.e. z value for the 
        # output layer
        for zn in range(self.n_hidden_layers + 1)[::-1]:
            # For output layer
            if zn == self.n_hidden_layers:
                self.J[zn] = np.mean(self.activ_fns[zn](self.layers[zn], deriv=True), axis=0)
            # Hidden layer
            else:
                self.J[zn] = self.J[zn+1]@(self.weights[zn+1].T@np.mean(self.activ_fns[zn](self.layers[zn], deriv=True), axis=0))

        return self.J
    
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
            self.back_propagate(subset_X, subset_Y, pred, lr)

            
            # Calculate loss
            pred = self.feed_forward(X_test)
            # breakpoint()
            loss[n] = np.sum(np.mean(self.loss_fn(Y_test, pred, deriv=False), axis=0))

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
    
    def back_propagate(self, X: np.array, Y: np.array, pred: np.array, lr: float):
        """
        Updates weights and biases using backpropagation

        Args:
            X (np.array): data matrix
            Y (np.array): Actual labels for the dataset (n rows, 1 column)
            pred (np.array): Predicted labels for the data (n rows, 1 column)
            lr (float): learning rate
        """

        dL_dyhat = np.mean(self.loss_fn(Y, pred, deriv=True), axis=0)

        # Go through and update weights and biases for
        # each layer
        for n in range(self.n_hidden_layers + 1)[::-1]:
            dL_dzn = dL_dyhat@self.J[n]
            self.dLdz[n] = dL_dzn
            # Not input layer
            if n > 0:
                H = self.activ_fns[n-1](self.layers[n-1])
                # breakpoint()
                self.weights[n] += -lr*np.mean([np.outer(hi, dL_dzn) for hi in H], axis=0)
            # Input Layer
            else:
                self.weights[n] += -lr*np.mean([np.outer(xi, dL_dzn) for xi in X], axis=0)
            self.biases[n] += -lr*dL_dzn
        
        return

    



def normalize_data(data: pd.DataFrame, label_col: str = "LABEL", OHE = False):
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
        OHE (bool, optional): whether to do one hot encoding for Y

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

    # DO OHE
    if OHE:
        Y_OHE = np.zeros((Y.shape[0], len(label_map)), dtype=int)
        for i in range(Y.shape[0]):
            Y_OHE[i, Y[i]] = 1
        Y = Y_OHE

    return X, Y


def plot_confusion_matrix(Y: np.array, pred: np.array, labels=[], savename=""):
    """
    Convenience function for generating a confusion Matrix

    Args:
        Y (np.array): Actual labels for the dataset (n rows, 1 column)
        pred (np.array): Predicted labels for the data (n rows, 1 column)
        labels (list of str): class labels
        savename (str, optional): File to save plot to. If none is given shows figure.
                                    Defaults to "".

    Returns:
        confusion matrix
    """
    # Figure out predicted class -- infer from Y and pred the number of classes
    if Y.shape[1] > 1:
        Y_labels = np.zeros(Y.shape[0], dtype=int)
        pred_labels = np.zeros_like(Y_labels)
        for i in range(Y.shape[0]):
            Y_labels[i] = np.argmax(Y[i])
            pred_labels[i] = np.argmax(pred[i])
    else:
        Y_labels = Y
        pred_labels = (Y >= 0.5).astype(int)
    cm = confusion_matrix(Y_labels, pred_labels)
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    if not labels:
        labels = np.arange(max(Y.shape[1], 2))
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    
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
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss (LR = {lr})")
    if savename != "":
        plt.savefig(savename)
        plt.close(f)
    else:
        plt.show()



if __name__ == "__main__":


    ##################################### PART 1

    # Initialize network and load data
    # data = pd.read_csv("A2_Data_EliWeissler.csv")
    # # data = pd.read_csv("A1_Data_EliWeissler.csv")
    # X, Y = normalize_data(data)

    # input_size = 3
    # hidden_layers = [4]
    # output_size = 1
    # activation_fns = [sigmoid, sigmoid]
    # loss_fn = loss_MSE
    # random_initialize = True
    # network = NeuralNetwork(input_size, output_size, hidden_layers,
    #                         activation_fns, loss_fn, random_initialize=random_initialize)
    
    # # Train network
    # epochs = 10000
    # lr = 0.1
    # batch_size = 1
    # loss = network.train(X, Y, X, Y, epochs=epochs, lr=lr, batch_size=batch_size)

    # # Predict
    # pred = network.feed_forward(X)

    # # Plot Titles
    # err_plot = "err.png"
    # conf_mat = "conf.png"

    # plot_confusion_matrix(Y, pred, conf_mat)
    # plot_loss(loss, err_plot)



    
    
    ##################################### PART 2
    data = pd.read_csv("A2B_Data_EliWeissler.csv")
    # data = data[data.label != "blue"]
    X, Y = normalize_data(data, OHE=True)

    # Initialize network and load data
    input_size = 3
    # hidden_layers = [8, 6, 4, 2]
    # activation_fns = [sigmoid, ReLU, sigmoid, ReLU, softmax]
    hidden_layers = [2]
    activation_fns = [ReLU, softmax]
    output_size = 4
    loss_fn = loss_CCE
    random_initialize = True
    network = NeuralNetwork(input_size, output_size, hidden_layers,
                        activation_fns, loss_fn, random_initialize=random_initialize)
    
    # Train network
    epochs = 1000
    lr = 1
    batch_size = 9
    loss = network.train(X, Y, X, Y, epochs=epochs, lr=lr, batch_size=batch_size, check_progress=1000)

    # Predict and plot
    pred = network.feed_forward(X)

    plot_confusion_matrix(Y, pred, savename="conf_mat.png", labels=["Green", "Red", "Blue", "Black"])
    plot_loss(loss, lr, savename="loss.png")