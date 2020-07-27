# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:35:35 2020

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


#comparison of both sets of data.
Emp_left = df.groupby('Emp_left')
Emp_left= Emp_left.mean()

#characteristics of the dataset
head = df.head()
tail = df.tail()
info = df.info()
describe = df.describe()

