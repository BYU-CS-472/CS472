import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PerceptronClassifier(BaseEstimator,ClassifierMixin):
    
    #### you can add your own params for debug purposes but you need to have
        # LR --- Learning rate
        # deterministic -- Deterministic is for grading purposes. When Determinisitic != -1 shuffling is turned off and you train the model for the number of epochs e.g. if determinsitic = 10 then you train the model for 10 epochs without shuffling
        # shuffle do I shuffle or do I not.
    def __init__(self,LR=.1,deterministic=-1,shuffle=True):
        self.LR = LR
        self.shuffle = shuffle
        self.deterministic = deterministic
        if self.deterministic != -1:
            self.shuffle = False
    ### Fit is Training the model
    def fit(self,data,labels):
        pass
    ### After the model is trained you can called predict and it makes a prediction on the given input
    def predict(self,test):
        pass
    ### you see how well the model does on the given data and labels/targets. Returns Accuracy.
    def score(self,data,target):
        pass
    ### Given for your convenience
    def shuffle_data(self,data,labels):
        temp = np.hstack((data,labels))
        np.random.shuffle(temp)
        data,labels = temp[:, :data.shape[1]], temp[:, data.shape[1]:]
        return data,labels
    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def getWeights(self):
        pass
