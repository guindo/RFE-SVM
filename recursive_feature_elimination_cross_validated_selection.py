# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:54:39 2020

@author: guindo
"""
import numpy as np
# import matplotlib.pyplot as plt

import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
# from numba import jit, cuda



import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


    
dataset = pd.read_csv('fulldata1.csv')

X = dataset.iloc[:, 0:2151].values
y = dataset.iloc[:, 2151:2152].values


min_max_scaler = MinMaxScaler()
X1= min_max_scaler.fit_transform(X)
y=y.ravel();
    # Build a classification task using 3 informative features
    
    # Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    
rfecv = RFECV(estimator=svc, step=3, cv=StratifiedKFold(10),
                  scoring='accuracy')
rfecv.fit(X1, y.ravel())
print("Optimal number of features : %d" % rfecv.n_features_)
p=rfecv.n_features_


asd=rfecv.get_support()
# X_dataframe = pd.DataFrame(X)
X_dataframe = pd.read_csv('fulldatanolab.csv')

# data1 = rfecv.transform(X_dataframe)





selected_feat =X_dataframe.columns[asd]



# X_dataframe1 = pd.DataFrame(X).T  
X_dataframe1 = X_dataframe.T

#real feature selectioned
newdata2 = X_dataframe1[asd].T

#changing label to dataframe sinon concat impossible
# label = pd.DataFrame(y)

# newdata2 = pd.concat([newdata2, label], axis = 1)
gfg_csv_data = newdata2.to_csv('importantfeaturewithrecursive.csv', index = True) 















   
X_dataframe = pd.DataFrame(X)

bestfeatures=X_dataframe.columns[rfecv.support_]
originaldata=X_dataframe.columns
print('Best features :', X_dataframe.columns[rfecv.support_])
    
ppp=rfecv.support_
X_dataframe = pd.DataFrame(X).T  

#real feature selectioned
newdata2 = X_dataframe[ppp].T
    
    # Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
    