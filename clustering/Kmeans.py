import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClassifier(BaseEstimator,ClusterMixin):

    def __init__(self,k=3): ## add parameters here
        """
        Args:
            k = how many final clusters to have
        """
        self.k = k

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        return self
    def save_clusters(self,filename):
        """
            f = open(filename) 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,seperator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
        """
