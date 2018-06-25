# Importing the libraries
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#load data
iris=load_iris()
dir(iris)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
iris_train, iris_test,target_train, target_test = train_test_split(iris.data, iris.target,test_size = 0.3,random_state = 0)

print(iris.DESCR)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

print(iris_train)
print(iris_test)
print(target_train)
print(target_test)

#calling algo of descisiontreeclassifier
clf_dec = DecisionTreeClassifier()
trained_dec = clf_dec.fit(iris_train, target_train)
output_dec = trained_dec.predict(iris_test)
print(output_dec)

#calling algo of KNNclassifier
clf_knn = KNeighborsClassifier(n_neighbors= 5)
trained_knn = clf_knn.fit(iris_train, target_train)
output_knn = trained_knn.predict(iris_test)
print(output_knn)

#checking % accuracy for decisiontree
from sklearn.metrics import accuracy_score
checkpct_dec = accuracy_score(target_test, output_dec)
print(checkpct_dec)

#checking % accuracy for KNN
from sklearn.metrics import accuracy_score
checkpct_knn = accuracy_score(target_test, output_knn)
print(checkpct_knn)

#Visualitation of graph
tree.export_graphviz(clf, out_file="tree.dot", max_depth=None, feature_names=iris.feature_names, 
                 filled=True,rounded=True, precision=3)

