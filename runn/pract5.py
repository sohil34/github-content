import numpy as np
import pandas as pd
#Import dataset
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#Load dataset
wine = datasets.load_wine()
#print (wine) #if you want to see the data you can print data
#print the name of the 13 features
print("Features: ", wine.feature_names)
#print the label type of wine
print("Labels: ", wine.target_names)
X=pd.DataFrame(wine['data'])
print(X.head())
print(wine.data.shape)
#print the wine labels (0:Class_0, 1:class_2, 2:class_2)
y=print (wine.target)
#import train_test_split function
from sklearn.model_selection import train_test_split
#split dataset into training set and test set.
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size
=0.30,random_state=10)
#import gaussian naive bayes model.
from sklearn.naive_bayes import GaussianNB
#create a gaussian classifier
gnb = GaussianNB()
#train the model using the training sets
gnb.fit(X_train,y_train)
#predict the response for test dataset
y_pred = gnb.predict(X_test)
print(y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#confusion matrix

cm=np.array(confusion_matrix(y_test,y_pred))
print(cm)