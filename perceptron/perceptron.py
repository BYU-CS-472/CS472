import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

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

    def fit(self, X, y, initial_weights=None, stop_thresh=0.01):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        num_features = len(x[0])
        if not initial_weights:
            self.W = self.initialize_weights(num_features)
        else:
            self.W = initial_weights
        b = np.ones(num_features)
        X_b = np.append(X, b, axis=1)

        score = self.score(X, y)
        while True:
            for i in range(len(X)):
                z = X_b[i] * self.W
                if z > 0:
                    z = 1
                else:
                    z = 0
                change_coeff = (y[i] - z) * self.lr
                change_W = X[i] * change_coeff
                self.W += change_W
            
            new_score = self.score(X, y)
            if score - new_score < stop_thresh:
                break
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
        pass

    def initialize_weights(self, size):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:
            W (array-like): set of weights of specified size + 1 (to include bias)
        """
        W = np.zeros(size + 1)
        return W

    def score(self, X, y):
        num_features = len(X[0])
        b = np.ones(num_features)
        X_b = np.append(X, b, axis=1)
        z = X_b * self.W
        z = np.where(z > 0, 1, 0)
        diff = np.absolute((z - y))
        acc = 1 - (np.sum(diff) / len(diff))

        return acc

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        pass
