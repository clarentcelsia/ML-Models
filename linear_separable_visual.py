# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:53:32 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.svm import SVC

from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


from sklearn.datasets import make_blobs


# Linear separable visualization 

def svm_decision_function(model, ax=None, plot_support=True):
 
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid  
    x = np.linspace(xlim[0].min(), xlim[1].max(), 20)
    y = np.linspace(ylim[0].min(), ylim[1].max(), 20)
    Y, X = np.meshgrid(y, x)
    
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='g');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def params_c():
    
    # make_blobs
      # n_samples = total samples
      # n_features = X
      # centers = y
      # shuffle = shuffle the samples
      # cluster_std = standard deviation of samples
    X, y = make_blobs(n_samples=50, n_features= 2, centers=2, random_state=0, cluster_std=0.7)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for axis, C in zip(ax, [1.0, 10.0, 0.1]):
        print(C)
        model = SVC(kernel='linear', C=C).fit(X_train, y_train)
        
        # Plot training
        axis.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='Set1')
        svm_decision_function(model, axis)
        axis.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
        axi.set_title('C = {0:.1f}'.format(C), size=14)

                
    
def line_visual():
    
    data = datasets.load_iris()
    X = data.data[:, :2]  # we only take the first 2 features.
    Y = data.target
    
    #Binary classification problem
    X = X[np.logical_or(Y==0,Y==1)]
    Y = Y[np.logical_or(Y==0,Y==1)]
    
    
    # Dataframe of Iris
    irisDF = pd.DataFrame(data = data.data, columns = data.feature_names)
    irisDF['Target'] = pd.DataFrame(data.target)
    print("\nIris dataframe: \n",irisDF.head(8)) # print dataset only first 8 rows 
    
    
    # Data visualization
      # scatter_matrix(frame, figure_size, ..)
    scatter_matrix(irisDF.iloc[:,0:4], figsize=(15,10))
      #  4x4 matrices 

    # Build model
    svm = SVC(kernel='poly')
    svm.fit(X, Y)
    
    # Assume here wants to compare 2 features: sepal length n sepal width
      # x as sepal length
      # y as sepal width
    #x = irisDF.iloc[:, 0]
    #y = irisDF.iloc[:, 1]
 
    iris_names = data.target_names
    colors = ['red', 'blue', 'green']
    label = (data.target).astype(np.int)
    
    plt.figure(figsize=(15,10))
    #plt.scatter(x, y, c = 'red', marker= 's', alpha=0.5) # in one color image
    
    # Get different color of classes
      # range(len(irish_names)): return the number of items in irish_names: 3 
        # while the target(label) is (0,1,2) so the color is (red, blue, green) 
    for i in range(len(iris_names)):
        separate_df = irisDF[irisDF['Target'] == i]
        separate_df = separate_df.iloc[:,[0,1]].values
        print(separate_df, "\n")
        plt.scatter(separate_df[:, 0], separate_df[:, 1], label = iris_names[i]) 
        

    plt.scatter(separate_df[:, 0], separate_df[:, 1], s=50, cmap='Set1')    
    svm_decision_function(svm)    
    plt.title("sepal length vs sepal width")
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.legend()
    plt.grid()
    plt.show()
  
   
    
def plane_visual():
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we only take the first three features.
    Y = iris.target
 
    #Binary classification problem
    #X = X[np.logical_or(Y==0,Y==1)]
    #Y = Y[np.logical_or(Y==0,Y==1)]

    model = SVC(kernel='linear')
    clf = model.fit(X, Y)

    # The equation of the separating plane is ((w)svc.coef_[0], x) + b = 0.
    # Solve for w
    z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
    
    #Create grid
    xm, xM = X[:,0].min(), X[:, 0].max()
    ym, yM = X[:,1].min(), X[:, 1].max()
    x = np.linspace(xm, xM, 10)
    y = np.linspace(ym, yM, 10)
    x, y =np.meshgrid(x, y)
    
    #3D Visualize
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'sr')
    ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'ob')
    ax.plot_surface(x, y, z(x,y))
    ax.view_init(30, 60)
    plt.grid()
    plt.show()
       
    
if __name__ == "__main__":
    params_c()
    line_visual()
    plane_visual()