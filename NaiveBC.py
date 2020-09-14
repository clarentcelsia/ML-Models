# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:52:09 2020

@author: user
"""

# NAIVE BAYES
# HOW TO DO CLASSIFICATION USING 3 MODELS OF NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


filename = 'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Image Classification\\dataset\\diabetes.csv'
        
# IMPORT DATASET
def Datasets():
    """
    In this section,
    read dataset, split the dataset into 3 sets : train_set, test_set, and valid_test
        train_set : to build model
        test_set : for testing model
        valid_test : to cross check back the model 

    """
    # READ CSV 
    diabetes = pd.read_csv(filename)
       #print(diabetes) : to see the table
        # DEPENDENT VARs
          #X = diabetes.iloc[:, :8] # entire rows and take first 8 columns 
        # INDEPENDENT VAR : Outcome
          #y = diabetes['Outcome']
    
    # Splitting the data into training, testing and validating set for cross checking 
    train_set, temp_test_set = train_test_split(diabetes, test_size = 0.2, random_state = 0)
    print("training set: ", train_set.shape)
    print("temp_testing set: ", temp_test_set.shape)
    
    test_set, validation_set = train_test_split(temp_test_set, test_size = 0.2, random_state = 0) 
    print("testing set: ", test_set.shape)
    print("validating set: ", validation_set.shape)
    
    # Plot the relationship between each 2 vars to spot anything incorrect
    train_plots = train_set.describe()
    print(train_plots)
    
    # Get label for label
    train_labels = train_set['Outcome']
    test_labels = test_set['Outcome']
    valid_labels = validation_set['Outcome']
    
    # Drop label for features
    train_plots.pop("Outcome")
    train_set.pop("Outcome")
    test_set.pop("Outcome")
    validation_set.pop("Outcome")
    
    # Visualize the plots before the symmetrization
    sns.pairplot(train_plots[train_plots.columns], diag_kind='kde') #diag_kind = 'reg'
  
    # Symmetrical matrix : adding up  with its transpose value
      # finding the mean, standard_deviation, min, max, count, ... of the features
    symmetric_train_plots = train_plots.transpose()

    # Visualize the plots after the symmetrization
    sns.pairplot(symmetric_train_plots[symmetric_train_plots.columns], diag_kind='kde') #diag_kind = 'reg'
  
    return symmetric_train_plots,\
           train_set,\
           test_set,\
           validation_set,\
           train_labels,\
           test_labels,\
           valid_labels

    
def normalize_form(dataset, symmetrical_matrix):
    """
    Normalizing: for each element divided by sum of spatial elements
    """
    return (dataset - symmetrical_matrix['mean'])/symmetrical_matrix['std']

def normalizing():
    symmetric_trainset, train_set, test_set, validation_set, train_label, test_label, valid_label = Datasets()
    
    normed_train = normalize_form(train_set, symmetric_trainset)
    normed_test = normalize_form(test_set, symmetric_trainset)
    normed_valid = normalize_form(validation_set, symmetric_trainset)
    
    """
    np.savetxt("normed_train_features.csv", normed_train, delimiter=',')
    np.savetxt("normed_train_labels.csv", train_label, delimiter=',')
    np.savetxt("normed_test_features.csv", normed_test, delimiter=',')
    np.savetxt("normed_test_labels.csv", test_label, delimiter=',')
    np.savetxt("normed_valid_features.csv", normed_valid, delimiter=',')
    np.savetxt("normed_valid_labels.csv", valid_label, delimiter=',')
    """
    
    return normed_train

    
def NaiveBayes():
    #GET DATA 
    normed_train_features = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_train_features.csv'
    normed_train_labels = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_train_labels.csv'
    
    normed_test_features = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_test_features.csv'
    normed_test_labels = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_test_labels.csv'
    normed_valid_features = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_valid_features.csv'
    normed_valid_labels = 'D:\\ClarentiCelsia\Personal\\SKRIPSI\\Image Classification\\dataset\\normed_valid_labels.csv'
    
    train_features = pd.read_csv(normed_train_features)
    train_labels = pd.read_csv(normed_train_labels)
    
    test_features = pd.read_csv(normed_test_features)
    test_labels = pd.read_csv(normed_test_labels)
    valid_features = pd.read_csv(normed_valid_features)
    valid_labels = pd.read_csv(normed_valid_labels)
    
    X_train = train_features.iloc[:, :]
    y_train = train_labels.iloc[:, 0]
    
    X_test = train_features.iloc[:, :]
    y_test = train_labels.iloc[:, 0]
    X_valid = valid_features.iloc[:, :]
    y_valid = valid_labels.iloc[:, 0]
    
    # Build Model
      # train system(model) with the training set
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
 
    # Using testing set to predict the accuracy of the model 
      # Can system detect the testing set accurately?
    y_pred = gnb.predict(X_test)  
    print("what system predicts toward the testing set of gnb: ", y_pred)
      
    accuracy = accuracy_score(y_test, y_pred)    
    print("The accuracy of GNB: ", accuracy)
    
    # Figure out the confusion matrix suchas recall, precision,..
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print("The confusion matrix: ", confusionmatrix)
    
    print("\n")
    # Evaluating by using Cross val on Training Set & Testing set
    test_scores = cross_val_score(gnb, X_test, y_test, scoring='accuracy', cv=5)
    print("Accuracy of testing set by using cross val : ",test_scores)
    print("Mean of test scores: ", test_scores.mean())
      
    
    # Cross checking back
    # Check the probability of prediction value
    predict_proba_value = gnb.predict_proba(X_valid) 
    y_pred_valid = gnb.predict(X_valid)  
    
    print("what value of prediction probability so system can determine which class of them: ", predict_proba_value)
    print("what system predicts toward the validating set of gnb: ", y_pred_valid)
  
    
  
   
if __name__ == "__main__":   
    NaiveBayes()
    
    print("\n")
        



"""
Naive Bayes models : 
    - Gaussian NB
    - Multinomial NB
    - Bernoulli NB
    
    https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
"""
    