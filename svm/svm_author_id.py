#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = SVC(kernel='rbf', C=10000.)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t1 = time()
clf.fit(features_train, labels_train)
print('Training Time = ', round(time()-t1, 3))

t2 = time()
pred = clf.predict(features_test)
print('Prediction Time = ', round(time()-t2, 3))
# print('Predictions 10th = {} , 26th = {}, 50th = {} '.format(pred[10], pred[26], pred[50]))
pred_chris = [p for p in pred if p > 0]
print('predictions for chris = ', len(pred_chris))
t3 = time()
accuracy = accuracy_score(pred, labels_test)
print('Accuracy = ', accuracy)
print('Accuracy Time = ', round(time()-t3, 3))
#########################################################


