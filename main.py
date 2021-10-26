#lab 1 artintmet

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

X, y = datasets.make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_repeated=0,
    n_redundant=0,
    flip_y=.05,
    random_state=1410,
    n_clusters_per_class=1
)

# X is a matrix
print(X.shape, y.shape)

'''
plt.figure(figsize=(5,2.5))
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.tight_layout()
plt.savefig('scatter.png')'''

dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)

#stores dataset to csv format
np.savetxt(
    "dataset.csv",
    dataset,
    delimiter=",",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"])

#reads csv file to create a dataset
dataset = np.genfromtxt("dataset.csv", delimiter=",")
X, Y = dataset[:,:-1], dataset[:,-1].astype(int)

