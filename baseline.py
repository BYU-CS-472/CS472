### This is complete ###
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BaselineClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,param1=True): ## add parameters here
        self.param1 = param1

    #This is the function to Train your weights.
    #In the Baseline Classifier it simply learns what the Most Common Value is.
    def fit(self,data,labels):
        flat = labels.flatten().astype(np.int16)
        bins = np.bincount(flat) ### bin the data aka count each class 
        self.most_common = np.argmax(bins) ### grab the class with the highest count

    #Given Novel input it predict an output
    #each row in data is a row in the dataset or a single data point
    def predict(self,data):
        h,w = data.shape
        prediction = np.full((h,1),self.most_common) # always guess the most common
        return prediction

    #Returns the Mean(Accuracy) score given input data and labels
    def score(self,data,labels):
        h,w = data.shape
        predictions = self.predict(data)
        diff = labels.reshape(-1,1) - predictions.reshape(-1,1)
        diff[diff != 0] = 1
        incorrect = np.sum(diff)
        return 1 - (incorrect / h)
