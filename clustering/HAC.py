import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClassifier(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
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
    def print_clusters(self):
        """
            Used for grading.
            Print(# of clusters)
            print(Total SSE of all Clusters)
            print() <-- space in between
            for each cluster:
                print(Centroid_Value) <-- usually HAC doesn't have a centroid but for grading purposes we require a centroid.
                print(# of instances for cluster)
                print(SSE)
                print() <-- space in between clusters

        """


