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
            float: The accuracy
        """

        # This calls the parent class score function; for our labs, you must code your own score function
        return super().score(X,y)


### This file is meant to be imported. However, if you want something to happen when you run it directly,
#   e.g. "python baseline.py", you can put that in the block below. This can be a convenient place to put
#   a test case.

if __name__ == "__main__":
    my_baseline_classifier = BaselineClassifier()
    train_data = np.array(range(0, 9)).reshape(3, 3) # create 3x3 array, 0 through 8
    labels = np.array([0,0,5])  # create labels
    test_data = np.array([[1,1,1],[1,4,4]])

    my_baseline_classifier.fit(train_data, labels) # run fit; should find the modal (most common) label (0)
    prediction = my_baseline_classifier.predict(test_data) # doesn't matter what my data looks like, baseline will always return 0
    print(prediction)

