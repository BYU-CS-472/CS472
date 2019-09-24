import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pdb

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None, stop_thresh=0.01, num_stopping_rounds=5, deterministic=-1):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
            stop_thresh (float): threshold of desired accuracy change before training stops
            num_stopping_rounds (int): number of rounds that need to pass without improvement beyond stop_thresh to stop
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        num_features = len(X[0])
        if not initial_weights:
            self.W = self.initialize_weights(num_features)
        else:
            self.W = initial_weights
        num_instances = len(X)
        b = np.ones((num_instances, 1))
        X_copy = np.append(X, b, axis=1)
        y_copy = y.copy()

        score = self.score(X, y)
        num_bad_rounds = 0
        rounds_left = 0
        if deterministic > 0:
            rounds_left = deterministic

        while True:
            if self.shuffle:
                X_copy, y_copy = self._shuffle_data(X_copy, y_copy)
            for i in range(len(X_copy)):
                z = np.dot(X_copy[i], self.W)
                z = np.where(z > 0, 1, 0)
                change_coeff = (y_copy[i] - z) * self.lr
                change_W = X_copy[i] * change_coeff
                change_W = change_W.reshape(-1, 1)
                self.W += change_W

            if deterministic > 0:
                rounds_left -= 1
                if rounds_left <= 0:
                    break
            else:
                new_score = self.score(X, y)
                if score - new_score < stop_thresh:
                    num_bad_rounds += 1
                    if num_bad_rounds >= num_stopping_rounds:
                        break
                else:
                    num_bad_rounds = 0
                score = new_score

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        num_instances = len(X)
        b = np.ones((num_instances, 1))
        X_b = np.append(X, b, axis=1)
        z = np.dot(X_b , self.W)
        z = np.where(z > 0, 1, 0)
        return z

    def initialize_weights(self, size):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:
            W (array-like): set of weights of specified size + 1 (to include bias)
        """
        W = np.zeros((size + 1, 1))
        return W

    def score(self, X, y):
        num_instances = len(X)
        b = np.ones((num_instances, 1))
        X_b = np.append(X, b, axis=1)
        z = np.dot(X_b , self.W)
        z = np.where(z > 0, 1, 0)
        diff = np.absolute((z - y))
        acc = 1 - (np.sum(diff) / len(diff))

        return acc

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        tmp = np.append(X, y, axis=1)
        np.random.shuffle(tmp)
        X = tmp[:, :-1]
        y = tmp[:, -1].reshape(-1, 1)
        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.W
