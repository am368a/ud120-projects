#!/usr/bin/python

import math
from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

# features_train = features_train[:len(features_train)/10]
# labels_train = labels_train[:len(labels_train)/10]

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
# k = math.sqrt(len(features_train))
k = 13
print 'K value = {}'.format(k)

clf = KNeighborsClassifier(n_neighbors=k, p=3)

t1 = time()
clf.fit(features_train, labels_train)
print 'Time to train = {}'.format(round(time()-t1, 3))

t2 = time()
pred = clf.predict(features_test)
print 'Time to predict = {}'.format(round(time()-t2, 3))

acc = accuracy_score(pred, labels_test)
print 'Accuracy = {}'.format(acc)



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
