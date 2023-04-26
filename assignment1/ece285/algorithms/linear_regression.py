"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # make self.w as ((D+1)xC))
        self.w = self.w.T

        # We want to consider the vectorized equations 
        # y = XW ; where X is (Nx(D+1)) and W is ((D+1)xC) and y is NxC
        # append 1's to the last column of X_train
        # no need to do: already done in preprocessing X_train = np.hstack((X_train, np.ones((N, 1))))
        y_pred = X_train @ self.w

        # error (y_pred - y_train) should be NxC
        # one-hot encode y_train
        y_train = one_hot_encode(y_train)
        error = y_pred - y_train

        # calculate gradient; gradient should be DxC
        gradient = 1.0/N*((X_train).T@(y_pred-y_train) + self.weight_decay*self.w)

        # update weights
        for i in range(self.epochs):
            self.w = self.w - self.lr*gradient

        # return weights in the same shape as input weights
        return self.w.T

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        y_pred = X_test @ self.w
        y_pred = np.argmax(y_pred, axis = 1)
        return y_pred


def one_hot_encode(y_train):

    """Use to one hot encode y_train
    
    Paramerters:
        y_train: a numpy array of shape (N,) containing training labels
    
    Returns:
        y_train: a numpy array of shape (N, C) containing training labels
    """
    
    N = y_train.shape[0]
    C = np.max(y_train) + 1
    y_train = np.eye(C)[y_train]
    return y_train
