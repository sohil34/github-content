import numpy as np
import pandas as pd
import urllib.request

url=urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    
names = ['sepal-length','sepal-width','petal-length','petal-width',"Class"]
dataset= pd.read_csv(url, names=names)
dataset.head()
x=dataset.drop('Class',axis=1)
y=dataset['Class']
x.head()
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train1=sc.fit_transform(x_train)
x_test1= sc.transform(x_test)
y_train1 = y_train
y_test1 = y_test

from sklearn.decomposition import PCA
pca=PCA()
x_train1=pca.fit_transform(x_train1)
x_test1=pca.transform(x_test1)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
x_train1 = pca.fit_transform(x_train1)
x_test1= pca.transform(x_test1)
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(x_train1, y_train1)
y_pred=classifier.predict(x_test1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print('Accuracy', accuracy_score(y_test,y_pred))