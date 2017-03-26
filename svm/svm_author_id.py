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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Creatig SVC.
# Tuning kernal and C params
# 	tried kernel with 'leanear' and 'rbf'
# 	and C=10, 100, 1000 and 10000
clf = SVC(kernel='rbf', C=10000.0)

# Sampling the data 1%
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

clf.fit(features_train, labels_train)

predictions = clf.predict(features_test)
#print("10th: ", predictions[10])
#print("26th: ", predictions[26])
#print("50th: ", predictions[50])


#count=0
#for prediction in predictions:
#	if prediction == 1:
#		count+=1
#
#print("Class 1 predictions: ", count);

score = accuracy_score(labels_test, predictions)
print "Score:", score

#########################################################


