from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np


def loadData(filename):
    X, d = [], []
    for line in open(filename).readlines():
        unpacked = line.rstrip().split(',')
        X.append([float(v) for v in unpacked[:-1]])
        if unpacked[-1] == "Iris-setosa":
            d.append([1, 0, 0])
        elif unpacked[-1] == "Iris-versicolor":
            d.append([0, 1, 0])
        elif unpacked[-1] == "Iris-virginica":
            d.append([0, 0, 1])

    X = np.matrix(X)
    d = np.matrix(d)
    return X, d


X_train, d_train = loadData("data/train.txt")
X_test, d_test = loadData("data/test.txt")
for i in range(3):
    p = linear_model.Perceptron()
    dn = np.ravel(d_train[:, i])
    p.fit(X_train, dn)
    y = p.predict(X_test)
    precision = precision_score(np.ravel(d_test[:, i]), y)
    recall = recall_score(np.ravel(d_test[:, i]), y)
    print("Output", i + 1, precision * 100, recall * 100)