### This is complete ###
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BaselineClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, example_parameter=True):
        """ Initialize class with chosen hyperparameters.
        """

        self.example_parameter = example_parameter

    def fit(self, X, y):
        """ This is the function to Train your weights. This Baseline Classifier
         just finds the mode of your data.

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
        """
        flat = y.flatten().astype(np.int16)
        bins = np.bincount(flat) ### bin the data aka count each class 
        self.most_common = np.argmax(bins) ### grab the class with the highest count

    def predict(self, X):
        """ Predict all classes for a dataset

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
        """
        h,w = X.shape
        prediction = np.full((h,1),self.most_common) # always guess the most common
        return prediction

    def score(self, X, y):
        """ Return accuracy of model on a given dataset.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:

        """
        h,w = X.shape
        predictions = self.predict(X)
        diff = y.reshape(-1, 1) - predictions.reshape(-1, 1)
        diff[diff != 0] = 1
        incorrect = np.sum(diff)
        return 1 - (incorrect / h)

if __name__ == "__test__":
    pass