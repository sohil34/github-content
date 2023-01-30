# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('DMVWrittenTests.csv')
X = dataset.iloc[:, [0,1]].values
Y = dataset.iloc[:,2].values
dataset.head(5)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
# Feature scaling
# is used to normalize the data within a particular range.
# as the data is widely varies.
# a small limit (-2,2).
# the score 96.51142588 is normalized to 1.55187648.
# are normalized to a smaller range.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
# Predicting the test set results.
y_pred = classifier.predict(X_test)
y_pred
# Confusion Matrix and Accuracy.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
from sklearn.metrics import accuracy_score
print ("Accuracy:", accuracy_score(Y_test, y_pred))
print(cm)