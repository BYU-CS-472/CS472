## Python, NumPy, and Arff Tutorial

This short tutorial demonstrates some basic functionality of Python and NumPy, as well as a custom Arff loader class.

### Modules

First, import the modules you will be using. For this example, arff.py should be in the same folder as your script.
```
import arff
import numpy as np
```

### Intro to Numpy Arrays
Numpy is Python's premier numerical array module. While Numpy arrays handle n-dimensions, this toolkit is tailored for .arff file data, which is generally 2-dimensional. The `.shape` property of a Numpy array will return a tuple of the array dimensions (rows, columns).

The array can also be "sliced" to obtain subsets:
```
my_array = np.asarray(range(0,25)).reshape(5,5)

# Get first two rows, from 4th column to the end
my_array[0:2, 3:]

# Get every other row, start at last column and go backward
my_array[::2, -1::-1]

# Get indices for all rows that have a 5 in them
row_idx = np.where(my_array==5)[0]
my_array[row_idx]

```

### Arff object class
Some of the datasets we use are stored in an Attribute-Relation File Format (.arff) file format. These can be loaded using the `arff` module:

```
arff_path = r"./test/datasets/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)
```

Here, `credit_approval` is an Arff object. The Arff object is mostly a wrapper around a 2D numpy array, which is stored as the 'data' Arff class variable, i.e. `credit_approval.data`. The Arff object also contains all the information needed to recreate the Arff file, including feature names, the number of columns that are considered "outputs" (labels), whether each feature is nominal or continuous, and the list of possible values for nominal features. Note that:

* The Arff object automatically encodes nominal/string features as integers. 
* The Arff object presently supports multiple labels, which are assumed to be the rightmost column(s). The number of labels should be passed explicitly with `label_count`, which is typically 1.
* `print(credit_approval)` will print the object as Arff text. Alternatively, a .arff style string can be obtained by taking `str(credit_approval)`.

The Arff object can also be sliced like traditional numpy arrays. E.g., the first row of data as a numpy array would be:

```
credit_approval[0,:]
```

Note that slicing this way returns a numpy 2D array, not an Arff object. To create a new Arff object that has been sliced, one can use:

```
# Get first 10 rows, first 3 columns
new_arff = Arff(credit_approxal, row_idx = slice(0,10), col_idx=slice(0,3), label_count=1)
```

Alternatively, one can use a `list` or `int` for either the `col_idx` or `row_idx`, but they should not be used for both simultaneously:

```
# Get rows 0 and 2, columns 0 through 9
new_arff = Arff(credit_approxal, row_idx = [0,2], col_idx=slice(0,10), label_count=1)

# Get row 1, all columns
new_arff = Arff(credit_approxal, row_idx = 1, label_count=1)

# Don't do this
new_arff = Arff(credit_approxal, row_idx = [2,3,8], col_idx = [1,2,3], label_count=1)
```

This ```new_Arff``` object will should copy the numpy array data underlying the original Arff. ```Arff.copy()``` can also be used to make a safe, deep copy of an Arff object.

To get the features of an Arff object as another Arff object, one can simply call:
```credit_approval.get_features()```

Similarly, for labels:
```credit_approval.get_labels()```

This may be helpful, since the Arff object has methods like:
* `unique_value_count(col)`: Returns the number of unique values for nominal variables
* `is_nominal(col)`: Returns true if the column is nominal
* `shuffle(buddy=None)`: Shuffles the data; supplying a buddy Arff with the same number of rows will shuffle both objects in the same order.

#### Other examples:
```
# Get 1st row of features as an ARFF
features = credit_approval.get_features(slice(0,1))

# Print as arff
print(features)

# Print Numpy array
print(features.data)

# Get shape of data: (rows, columns)
print(features.shape)

```

### Creating Learners

You will be creating classes for various machine learning models this semester, e.g. "MyPerceptron". This should inherit from `sklearn.linear_model` and override
at least the `fit()`, `predict()`, and `score()` functions. It should probably also have a constructor, i.e. `def __init__(self, argument1, argument2):` that can be used to initialize learner weights, hyperparameters, etc.

### Example using sci-kit learn
[insert here]

### Example cross-validation
[insert here]

### Graphing
A tiny graphing wrapper around matplotlib is included. See ```graph_tools.py```.

```
from toolkit import graph_tools
import matplotlib.pyplot as plt
import numpy as np

arff_path = r"./test/datasets/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)


## Graph a function using matplotlib
y_func = lambda x: 5 * x**2 + 1 # equation of a parabola
x = np.linspace(-1, 1, 100)
plt.plot(x, y_func(x))
plt.show()

## Scatter plot with 2 variables with labels coloring using graph_tools.py
x = credit_approval[:,1]
y = credit_approval[:,2]
labels = credit_approval[:, -1]
graph_tools.graph(x=x, y=y, labels=labels, xlim=(0,30), ylim=(0,30))
```

![alt text](https://raw.githubusercontent.com/cs478ta/CS478.github.io/master/toolkitPython/Scatter.png)
