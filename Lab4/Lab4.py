import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)

def KNN():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    estimator=KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    print("KNN Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)
    confusion_matrix=metrics.confusion_matrix(y_test, y_predict)
    return confusion_matrix

def DT():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    estimator=DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    print("DT Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)

def SGD():
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25,random_state=6)
    estimator=SGDClassifier()
    estimator.fit(x_train,y_train)
    y_predict=estimator.predict(x_test)
    accuracy_score=metrics.accuracy_score(y_test, y_predict)
    print("SGD Accuracy: %0.3f"%accuracy_score,end='\t')
    recall_score=metrics.recall_score(y_test, y_predict, average='macro')
    print("Recall: %0.3f"%recall_score)


print('COMP9517 Week 5 Lab - z5142254')
print()
test_size=x_test.shape[0]/(x_train.shape[0]+x_test.shape[0])
print('Test size = %0.2f'%test_size)
confusion_matrix=KNN()
SGD()
DT()
print()
print('KNN Confusion Matrix:')
print(confusion_matrix)