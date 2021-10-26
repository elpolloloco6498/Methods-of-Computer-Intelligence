import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

#dataset predictions

dataset = np.genfromtxt("dataset.csv", delimiter=",")
X, y = dataset[:,:-1], dataset[:,-1].astype(int)

#divide dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.3,
    random_state=42
)

clf = GaussianNB()
clf.fit(X_train, y_train)
class_probabilities = clf.predict_proba(X_test)
#print(class_probabilities)
predict = np.argmax(class_probabilities, axis=1)
print("True labels:     ", y_test)
print("Predicted labels:", predict)

#Evaluating the quality of our model
score = accuracy_score(y_test, predict)
print("Accuracy score:\n %.2f" % score)
#computing the confusion matrix
cm = confusion_matrix(y_test, predict)
print("Confusion matrix: \n", cm)




