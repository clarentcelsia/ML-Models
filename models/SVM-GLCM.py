# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:53:02 2020

@author: user
"""

# SVM - GLCM

import os
import skimage.io
import skimage 
import numpy as np
import pandas as pd
import cv2
import csv

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import greycomatrix ,greycoprops
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
from mlxtend.plotting import plot_decision_regions
from matplotlib.colors import ListedColormap

path = 'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\Resize_Leaf'

def GLCM(path, distance, angle):
    
    """
    d : distance 1, 2, ...
    angle : orientation 0, 45, 90, 135
    
    features : Contrast, Energy, Homogeneity, Correlation, ASM, Dissimilarity
    """
       
    # vars
    features = []
    label = 0

    for folder in os.listdir(path):
            
        label += 1
        file_path = "".join((path,"/",folder)) #'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\Resize_Leaf/leafA'
        
        for file in os.listdir(file_path):
            image_path = "".join((file_path, "/", file)) #'D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Proposal\\Resize_Leaf/leafA/leaf001.jpg'
            
            # Read Image
            img = skimage.io.imread(image_path, as_gray=True) # Convert img to grayscale
            
            # Converting... 
            img = skimage.img_as_ubyte(img) # Convert an image to 8-bit unsigned integer format.
            img = np.asarray(img, dtype="int32") # Convert the input to array, dtype = data_type, output = [0 0 1 0 ...]-> matriks nRows x nCols
            
            #print(img)
            #print("\n")
            # Calculate GLCM
               # A grey level co-occurrence matrix is a histogram of co-occurring greyscale values at a given offset over an image:
                 # image = integer type input image
                 # distance = jarak pasangan pixel
                 # angles = orientation/arah (0, 45, 90, 135)
                 # levels = The input image should contain integers dari np.array which is contain array_int
                         # number of grey-levels counted (typically 256 for an 8-bit image).
                         # img.max()+1 => img.max start from 0
                         # normed = normalize, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset
            glcm = greycomatrix(img, [distance], [angle], levels=img.max()+1, symmetric=True, normed=True)
            
            # Texture Props and Calculate feature statistic to analyze texture
            # [0][0] = 2D Array
              
             # contrast : intensity contrast between a pixel and its neighbor over the whole image.
             # energy :  sum of squared elements in the GLCM/ uniformitiy (keseragaman)
             # homogeneity : measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. 
             # correlation : measure of how correlated a pixel is to its neighbor over the whole image.
            contrast = greycoprops(glcm, prop='contrast')[0][0]
            energy = greycoprops(glcm, prop='energy')[0][0]
            homogeneity = greycoprops(glcm, prop='homogeneity')[0][0]
            correlation = greycoprops(glcm, prop='correlation')[0][0]
            asm = greycoprops(glcm, prop='asm')[0][0]
            dissimilarity = greycoprops(glcm, prop='dissimilarity')[0][0]
            
            # Print Result
            if not contrast is None or not energy is None or not homogeneity is None or  not correlation is None or not asm is None or not dissimilarity is None:
                params = [label, contrast, energy, homogeneity, correlation, asm, dissimilarity]
                
                features.append(params)
      
        np.savetxt(("GLCM[%s][%s].csv" %(distance, angle)),
                   features, delimiter=',') # delimiter is for seperating columns
    
    print("GLCM extracting succeed!")
        
    
def load_glcm_csv(d, angle):
    csv = ('D:\\ClarentiCelsia\\Personal\\SKRIPSI\\Image Classification\\dataset\\GLCM[%s][%s].csv' %(d, angle))
           
    data = pd.read_csv(csv)
    
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    """
    # Visualize the dataset
    plt.figure()
    for i, j in enumerate(np.unique(y)):
       plt.scatter(X[y == j, 0], X[y == j, 1], c = ListedColormap(('red', 'green', 'blue', 'yellow', 'gray'))(i), label = j)
         
    plt.title('GLCM')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid()
    plt.show()
    """
    
    return X, y
    
def model(models, d, angle, C=None, kernel=None, visualize=False):
    
    X, y = load_glcm_csv(d, angle)
    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Feature scalling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if (models == 'svm' and kernel == None and C==None):
        kernel = 'linear'
        C=1
      
    # Build model
    svm = SVC(C=C, kernel=kernel).fit(X_train, y_train)
      
    if models == 'svm':
        #Accuracy of svm on testing and training set
        test_accuracy = svm.score(X_test, y_test)
        train_accuracy = svm.score(X_train, y_train)
        print('SVM Accuracy (training): ', train_accuracy)
        print('SVM Accuracy (testing): ', test_accuracy)
        
        # predict 
        y_pred = svm.predict(X_test)
        print('SVM predict: \n', y_pred, "\n")
        
        # prediction accuracy
          # system predict (features)x_test based on the training set
          # and detect upon y_test(y_actual)
        pred_accuracy = accuracy_score(y_test, y_pred)
        print(' Accuracy of SVM Prediction: {:.3f}'.format(pred_accuracy))
        
        # through confusion matrix
          # accuracy, precision, recall, f-score
        confusionmatrix = confusion_matrix(y_test, y_pred)
        classificationreport = classification_report(y_test, y_pred)
        print(' Accuracy of SVM Prediction using ConfusionMTX:\n', confusionmatrix, "\n")
        print(classificationreport)
        
        
        # by using crossval to evaluate the accuracy of training and testing set
          # estimator : svm.fit(X_train, y_train)
          # upon on the training set itself and testing set 
        train_scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=5)
        test_scores = cross_val_score(svm, X_test, y_test, scoring='accuracy', cv=5)
        print("Accuracy of training set by using cross val : ",train_scores)
        print("Accuracy of testing set by using cross val : ",test_scores)
        print("Mean of train scores: ", train_scores.mean())
        print("Mean of test scores: ", test_scores.mean())
  
        
    if visualize:
    
        X, y = load_glcm_csv(d, angle) #4 features
        X_samples = X[:, :2] #only take first 2 features
        y_samples = y
    
        # Splitting the dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=0)
    
        # Feature scalling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models.fit(X_train, y_train)
        
        # Visualizing the training set
          # Create 2D axis
            # X Axis (X1)
              # X_set[:,0].min() as start point
              # X_set[:,0].max() as end point
           # Y Axis (X2)
             # X_set[:,1].min() as start point
             # X_set[:,1].max() as end point
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),\
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))   

        plt.contourf(X1, X2, svm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                    alpha = 0.5, cmap = ListedColormap(('red', 'black', 'blue', 'yellow', 'gray')))
          
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
  
        for i, j in enumerate(np.unique(y_train)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s=20, c = ListedColormap(('red', 'green', 'blue', 'yellow', 'gray'))(i), marker='*', label = j)
        
        plt.title(" %s-GLCM[%s][%s]" % (models, d, angle))
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid()
        plt.show()
 
        # Visualizing the training set in 3D 
        rbf = np.exp(-(X_train ** 2).sum(1))
    
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter3D(X_train[:, 0], X_train[:, 1], rbf, marker='o', s=20, cmap='summer')
        ax.view_init(elev=0, azim=45)
        ax.set_title('3D form of GLCM[%s][%s]' %(d, angle))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
if __name__=="__main__":
    
    #GLCM(path, distance=1, angle=0)
    model('svm', d=1, angle=0, C=0.1, kernel='linear', visualize=True)
    
