# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 05:04:50 2020

@author: user
"""

# Another way to classify the images
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


filename = 'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Image Classification\\dataset\\diabetes.csv'

def Datasets():
    """
    just wants to split the dataset into 2 sets : train_set, test_set
        train_set : to build model
        test_set : for testing model
        valid_set : optional
        
    """
    # READ CSV 
    diabetes = pd.read_csv(filename)
       #print(diabetes) : to see the table
       
    # DEPENDENT VARs
    # X = diabetes.iloc[:, :8].values # entire rows and take first 8 columns 
    X = diabetes.iloc[:, [1,2]].values 
    # INDEPENDENT VAR : Outcome
    y = diabetes['Outcome'].values
    
    # Visualize the dataset before scalling
    sns.lmplot('Glucose', 'BloodPressure', data=diabetes, hue='Outcome',palette='Set1', fit_reg=False, scatter_kws={'s': 70})
    
    # Splitting the data into training, and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print("training set: ", X_train.shape)
    print("testing set: ", X_test.shape)
    
    # Feature scalling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
     
    # Build model
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    
    # Visualize support vectors of training set
    support_vectors = svm.support_vectors_

    plt.scatter(X_train[:,0], X_train[:,1])
    plt.scatter(support_vectors[:,0], support_vectors[:,1], color='blue')
    plt.title('Support Vectors')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
    
    # Visualize training & Testing set
      # Replace X_set, y_set according to the visualization 
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
           
    plt.title('SVM (training)')
    plt.xlabel('Glucose')
    plt.ylabel('Blood Pressure')
    plt.legend()
    plt.show()
    
    print("\n")

    # Predict
    y_pred = svm.predict(X_test)
    # Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of prediction: ", test_accuracy)
    
  
    # Figure out the confusion matrix suchas recall, precision,..
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print("The confusion matrix: ", confusionmatrix)
    
    print("\n")
    # Evaluating by using Cross val on Training Set & Testing set
    test_scores = cross_val_score(svm, X_test, y_test, scoring='accuracy', cv=5)
    print("Accuracy of testing set by using cross val : ",test_scores)
    print("Mean of test scores: ", test_scores.mean())
    
    print("\n")
    
    # Cross checking back
    # Check the probability of prediction value
    predict_proba_value = svm.predict_proba(X_test) 
    y_pred_valid = svm.predict(X_test)  
    
    print("what value of prediction probability so system can determine which class of them: ", predict_proba_value)
    print("what system predicts toward the validating set of svm: ", y_pred_valid)
  

   
if __name__ == "__main__":   
    Datasets()
    print("\n")

   
