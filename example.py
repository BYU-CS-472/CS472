from baseline.baseline import BaselineClassifier  # first baseline refers to the "baseline" folder, the second refers to the baseline.py
import numpy as np # we're just aliasing numpy with the standard np
from tools import graph_tools, arff

### Very simple baseline classifier example

my_baseline_classifier = BaselineClassifier()
train_data = np.array(range(0, 9)).reshape(3, 3)  # create 3x3 array, 0 through 8
labels = np.array([0, 0, 5])  # create labels
test_data = np.array([[1, 1, 1], [1, 4, 4]])

my_baseline_classifier.fit(train_data, labels)  # run fit; should find the modal (most common) label (0)
prediction = my_baseline_classifier.predict(test_data)  # doesn't matter what my data looks like, baseline will always return 0
print(prediction)

### Some Arff loading

arff_path = r"./data/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)

print(credit_approval.data) # the numpy array

## Graph 2 of the features as a scatter plot
x = credit_approval[:,1]
y = credit_approval[:,2]
labels = credit_approval[:, -1]
graph_tools.graph(x=x, y=y, labels=labels, xlim=(0,30), ylim=(0,30), title="Credit Plot")
