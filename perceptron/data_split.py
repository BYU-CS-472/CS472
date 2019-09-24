import numpy as np

def split(data, y=None, train_size=0.7):
    h, w = data.shape
    train_h = int(train_size * h)
    rows = np.arange(h)
    train_rows = np.random.choice(rows, size=train_h, replace=False)
    test_rows = np.array([x for x in range(h) if x not in train_rows])
    train = data[train_rows]
    test = data[test_rows]
    if y is None:
        return train, test
    train_y = y[train_rows]
    test_y = y[test_rows]
    return train, test, train_y, test_y
