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


    def fit(self, X, y, initial_weights=None):  # REQ
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        num_rows, num_cols = X.shape

        new_col = np.ones(num_rows, 1) # Bias added to X
        X = np.append(X, new_col, num_cols) # num_cols is the column that we want to append the 1's onto, not that it matters
        num_cols = X.shape[1]   # Num cols updated after addition of bias

        # self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        self.initial_weights = np.zeros(num_cols) # Weights initialized to 0
        weights = self.initial_weights
        delta_W = np.zeros(num_cols) # initializing delta ) to all zero's. They'll be updated later
        
        num_no_change_iters = 0

        while(num_no_change_iters < (num_rows * 10)):    # 10 eppochs of no change multiplied by num patterns
            for i in range(num_rows): # iterating over entire training set
                net = 0 # reset net after iterations
                output = 0
                for j in range(num_cols): # iterating over pattern
                    net = net + (X[i][j] * weights[j])
                if net > 0:
                    output = 1
                for j in range(num_cols):
                    delta_W[j] = self.lr *(y[i] - output) * X[i][j]  # Perceptron function to calculate delta W
                if delta_W == np.zeros(j):
                    num_no_change_iters = num_no_change_iters + 1
                else:
                    num_no_change_iters = 0

                weights = np.add(weights, delta_W)  # Apply delta W to weight vector using Numpy
            # First Eppoch completed
            # ImplementMe shuffle the data

        return self

    def predict(self, X):   # REQ
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
            
        return [0]

    def score(self, X, y):  # REQ
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        arry = np.append(X, y, 1) # axis is 1 to refer to columns
        np.random.shuffle(arry)


        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):  # REQ
        pass
