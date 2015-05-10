# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:33:55 2015

@author: david
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', index_col=0)

'''Data Preparation'''
#Changing Column Names
df.columns
df.rename(columns={'ReputationAtPostCreation':'Reputation', 'BodyMarkdown':'Body', 'PostCreationDate':'PostDate', 'OwnerCreationDate':'OwnerDate', 'OwnerUndeletedAnswerCountAtPostTime':'Answers', 'OpenStatus':'Status'}, inplace=True)

df.isnull().sum()


#Standardizing Dates...
df['PostDate'] = pd.to_datetime(df.PostDate)
df['OwnerDate'] = pd.to_datetime(df.OwnerDate)
df['PostClosedDate'] = pd.to_datetime(df.PostClosedDate)
#Standardizing Dates...

#Creating New Features
df['BodyLength'] = df.Body.apply(len)
df['TitleLength'] = df.Title.apply(len)

df['PostCount'] = 0
for item in df.sort('OwnerUserId'):
    

#Creating New Features


'''Data Preparation'''



'''Data Exploration'''
#PostId: Are more recent posts likely to be closed because the question has been answered before?
df.groupby(df.Status).PostId.describe()
df.groupby(df.Status).Answers.describe()
df.groupby(df.Status).OwnerDate.describe()
df.sort('OwnerDate')

pd.scatter_matrix(df)



'''PLOTTING TIME SERIES'''
dates = pd.date_range(df.PostDate.min(), df.PostDate.max(), freq='D')

print dates
print dates.shape

ts = pd.Series(df.Reputation.sum(), index=dates)
ts = ts.cumsum()
ts.plot()
'''PLOTTING TIME SERIES'''




'''Train Test Split'''
feat_cols = ['PostId', 'Answers', 'OwnerUserId']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[feat_cols], df.Status)

#LogReg Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_pred, y_test)
#LogReg Model

#LogReg proba Model



feat_cols = ['Reputation', 'Answers', 'OwnerUserId', 'BodyLength', 'TitleLength']
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[feat_cols], df.Status)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_prob = logreg.predict_proba(X_test)

from sklearn import metrics
metrics.log_loss(y_test, y_prob)

