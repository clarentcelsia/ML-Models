#!/usr/bin/env python
# coding: utf-8

# In[134]:


from ipywidgets import interact, fixed
from mpl_toolkits import mplot3d
from sklearn.datasets import make_circles

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

#jupyter notebook

X, y = make_circles(120, factor=.1, noise=.2)
rbf = np.exp(-(X ** 2).sum(1))
def svm_decision_function(model, ax=None, sv=True):

    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    hyperplane = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margin
    ax.contour(X, Y, hyperplane, colors='k',levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # plot support vectors
    if sv:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   marker='o' ,s=300, linewidth=.5, edgecolors = 'k',facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def nonlinear_2D():
    
    svm = SVC(kernel='linear').fit(X, y)
    
    plt.figure(figsize=(10,8))
    plt.scatter(X[:, 0], X[:, 1], marker='^', c=y, s=50, cmap='Set1')
    plt.title("Non linear separable")
    svm_decision_function(svm, sv=False)
    

def non_linear_3D(X, y, elev = [0,45], azim=(-90,90)):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1],rbf,marker='o', c=y, s=70, cmap='summer')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title('3D form of Non linear separable samples')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


nonlinear_2D()
#non_linear_3D(X,y)
interact(non_linear_3D, elev=[0,45], azim=(-90,90), X=fixed(X), y=fixed(y))

clf = SVC(kernel='rbf', C=10)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o',c=y, s=50, cmap='Set2')
svm_decision_function(clf)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




