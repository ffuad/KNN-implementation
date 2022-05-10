# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:45:08 2022

@author: mthff
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt

diabetes_data = load_diabetes()
diabetes = pd.DataFrame(diabetes_data.data)
diabetes_targets = pd.DataFrame(diabetes_data.target)

print(diabetes_data.feature_names)
print("\n")
print(diabetes.head())
print("\n")
print(diabetes_targets.head())

X = diabetes_data.data
y = diabetes_data.target

df = pd.read_csv('H:\diabetes.csv')
target = df['Outcome']
df.drop('Outcome',axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=100)
KNN.fit(X_train, y_train)

y_pred = KNN.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

error = []

#calculating error for K values between 1 and 100

for i in range(1,100):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  pred_i = knn.predict(X_test)
  error.append(np.mean(pred_i != y_test))
  #print(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1,100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')

