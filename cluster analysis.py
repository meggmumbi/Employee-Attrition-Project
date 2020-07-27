# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:15:51 2020

@author: paige
"""



#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', 
                        sheet_name='Employees who have left')
X = dataset.iloc[:, [1,2]].values 

# Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

# fitting kmeans to dataset
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='red')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='cyan')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green')

plt.title('Clusters of employees who left')
plt.xlabel('Satisfaction level')
plt.ylabel('Last Evaluation')
plt.savefig('clusters.png')
plt.legend()
plt.show()