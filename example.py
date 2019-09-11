from baseline.baseline import BaselineClassifier  # first baseline refers to the "baseline" folder, the second refers to the baseline.py
import numpy as np # we're just aliasing numpy with the standard np
from tools import graph_tools, arff
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model.perceptron import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Very simple baseline classifier example
my_baseline_classifier = BaselineClassifier()

train_data = np.array(range(0, 45)).reshape(15, 3)  # create 3x3 array, 0 through 8
train_labels = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5])  # create labels
test_data = np.array([[1, 1, 1], [1, 4, 4]])
test_labels = np.array([[5], [4]])

my_baseline_classifier.fit(train_data, train_labels)  # run fit; should find the modal (most common) label (0)
prediction = my_baseline_classifier.predict(test_data)  # doesn't matter what my data looks like, baseline will always return 0
print(f"Baseline prediction: {prediction}")

# Calculate score (accuracy)
test_score = my_baseline_classifier.score(test_labels, test_labels)
print(f"Test score: {test_score}")

# Calculate accuracy from predictions
test_score2 = accuracy_score(test_labels, prediction)

## Cross validation
cv_results = cross_val_score(my_baseline_classifier, train_data, train_labels, cv=3)
print(f"Cross-validation: {cv_results}")

## Grid search
# from sklearn.model_selection import GridSearchCV
# GridSearchCV(cv=5, error_score=...,
#        estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
#                      decision_function_shape='ovr', degree=..., gamma=...,
#                      kernel='rbf', max_iter=-1, probability=False,
#                      random_state=None, shrinking=True, tol=...,
#                      verbose=False),
#        iid=..., n_jobs=None,
#        param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
#        scoring=..., verbose=...)

### sklearn train / test split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)

### Some Arff loading
arff_path = r"./data/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)

print(credit_approval.data) # the numpy array

## Graph 2 of the features as a scatter plot
x = credit_approval[:,1]
y = credit_approval[:,2]
train_labels = credit_approval[:, -1]
graph_tools.graph(x=x, y=y, labels=train_labels, xlim=(0, 30), ylim=(0, 30), title="Credit Plot")
