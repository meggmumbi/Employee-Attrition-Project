# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:03:38 2020

@author: paige
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', 
                        sheet_name=['Existing employees', 'Employees who have left'])

df = pd.concat((dataset[frame] for frame in dataset.keys()), ignore_index=True)
x = df.iloc[:, [1,2,3,4,5,6,7,9]].values
y = df.iloc[:, 10].values

# Encoding independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[7])],
                                     remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

#encode dependent
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=0)

# Fitting Random Forest Classification to Training set
from sklearn.ensemble import RandomForestClassifier
classifier  = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Predicting Test set results
y_pred = classifier.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





















