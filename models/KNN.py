# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:38:02 2020

@author: user
"""

# KNN


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

filename = 'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Image Classification\\dataset\\diabetes.csv'

def Datasets():
    
    # READ CSV 
    diabetes = pd.read_csv(filename)
       #print(diabetes) : to see the table
       
    # DEPENDENT VARs
    X = diabetes.iloc[:, [1,2]].values # entire rows and take first 8 columns 
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
    knn = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski')
    knn.fit(X_train, y_train)
    
    # Visualize in 3d
    
    
    # Visualize training & Testing set
      # Replace X_set, y_set according to the visualization 
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
       plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
           
    plt.title('KNN (training)')
    plt.xlabel('Glucose')
    plt.ylabel('Blood Pressure')
    plt.legend()
    plt.show()
    
    print("\n")

    # Predict
    y_pred = knn.predict(X_test)
    # Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of prediction: ", test_accuracy)
    
  
    # Figure out the confusion matrix suchas recall, precision,..
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print("The confusion matrix: ", confusionmatrix)
    
    print("\n")
    # Evaluating by using Cross val on Training Set & Testing set
    test_scores = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=5)
    print("Accuracy of testing set by using cross val : ",test_scores)
    print("Mean of test scores: ", test_scores.mean())
    
    print("\n")
    
    # Cross checking back
    # Check the probability of prediction value
    predict_proba_value = knn.predict_proba(X_test) 
    y_pred_valid = knn.predict(X_test)  
    
    print("what value of prediction probability so system can determine which class of them: ", predict_proba_value)
    print("what system predicts toward the validating set of svm: ", y_pred_valid)
    print("\n")
    
    train_error(X_train, X_test, y_train, y_test)
  

   
def train_error(X_train, X_test, y_train, y_test):
    error = []
    for k in range(1, 10):
         knn = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski')
         knn.fit(X_train, y_train)
         
         y_pred = knn.predict(X_test)
         error.append(np.mean(y_pred != y_test))
         
    
    #np.savetxt("scatter_plot.csv", np.array(error), delimiter=',')
    # Visualize 
       # The best k that has min error
       # Choosing the right k value
    x_axis = range(1, 10)
    y_axis = error
    
    plt.figure(figsize=(15, 10))
    plt.plot(x_axis, y_axis, color='red', linestyle='solid', marker='o', markerfacecolor='green', markersize=8)
    
    # Displaying plot points (x,y)
    for x, y in zip(x_axis, y_axis):
        plt.text(x, y, '({:.2f}, {:.2f})'.format(x, y))
        
    plt.title('Error Rate')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
         

    
        
        
if __name__ == "__main__":   
    Datasets()
    print("\n")

   

"""
KNN:
    
"""