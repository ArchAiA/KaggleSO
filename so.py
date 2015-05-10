# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:52:24 2015

@author: david
"""

import pandas as pd

#Read in the data
df = pd.read_csv('train.csv', index_col=0)


'''Explore the Data'''
df.isnull().sum()
df.columns
df.head()

df.groupby(df.OpenStatus).describe()

#Half is open and half is closed (not a representative sample) - Generally 6% are closed
df.OpenStatus.value_counts()
#Some people post much more often than others
df.OwnerUserId.value_counts()


#Reputation
df.groupby(df.OpenStatus).ReputationAtPostCreation.mean() #Reputation matters
df.groupby(df.OpenStatus).ReputationAtPostCreation.describe()
df[df.ReputationAtPostCreation < 1000].ReputationAtPostCreation.hist()
df[df.ReputationAtPostCreation < 1000].ReputationAtPostCreation.hist(by=df.OpenStatus, sharey=True)
#Does this mean that if reputation is 13+ that there much less chance of the question being closed?




df.groupby(df.OpenStatus).OwnerUndeletedAnswerCountAtPostTime.mean() #Undeleted answer count matters
df.groupby(df.OpenStatus).OwnerUndeletedAnswerCountAtPostTime.describe() #Undeleted answer count matters

df.groupby(df.OpenStatus).OwnerUserId.mean() #People with lower user IDs had fewer closed questions
df.groupby(df.OpenStatus).OwnerUserId.describe() #People with lower user IDs had fewer closed questions
df.sort('OwnerUserId').OwnerCreationDate #UserIds do increase over time

#More recent posts more likely to be open
df.groupby(df.OpenStatus).PostId.describe()

df.ReputationAtPostCreation.describe()

#Some people post much more often than others
df.OwnerUserId.value_counts()
df[df.OwnerUserId == 466534]
df[df.OwnerUserId == 466534].groupby(df.OpenStatus).PostId.describe()
df[df.OwnerUserId == 466534].OpenStatus.sum()
#We should look more at people that post more often.  Do they get more closed questions than others?
#Do people that do not use proper upper/lower case have more closed questions.
#Do people with bad grammar have more closed questions?
#Do people with short questions have more closed questions?

#Looking at other users
df[df.OwnerUserId == 39677]
df[df.OwnerUserId == 34537]





#RENAMING COLUMNS
df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime': 'Answers'}, inplace=True)




#Checking if Title Length is Relevant: It's Not
df['TitleLength'] = df.Title.apply(len)
df.groupby(df.OpenStatus).TitleLength.describe()

#CHecking if Body Length is Relevant: It Kind of Is
df['BodyLength'] = df.BodyMarkdown.apply(len)
df.groupby(df.OpenStatus).BodyLength.describe()
df.BodyLength.hist(by=df.OpenStatus)



#Exploring Tags
#Do more tags == more closed questions, or more open questions?
len(df.Tag1.unique())
len(df.Tag2.unique())
len(df.Tag3.unique())
len(df.Tag4.unique())
len(df.Tag5.unique())

#We can use the agg() function to aggregate by the number of unique Tag1 occurrences
df.groupby(df.Tag1).OpenStatus.mean()
df.groupby(df.Tag1).OpenStatus.agg(['mean', 'count'])
df.groupby(df.Tag1).OpenStatus.agg(['mean', 'count']).sort('count')




df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
df.groupby(df.OpenStatus).NumTags.describe()






'''BUILDING A MODEL'''
feat_cols = ['ReputationAtPostCreation']
X = df[feat_cols]
y = df.OpenStatus

#Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Fitting the Model
from sklearn.linear_model import LogisticRegression #IMPORT
logreg = LogisticRegression()                       #INSTANTIATE
logreg.fit(X_train, y_train)                        #FIT

logreg.coef_ #The coefficient makes sense.  As reputation increases the log odds of the OpenStatus being 1 increases

#Predicting
y_pred = logreg.predict(X_test)

#Checking Accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)



#PREDICTING AGAIN USING PROBABILITIES
y_prob = logreg.predict_proba(X_test)[:, 1]
metrics.roc_auc_score(y_test, y_prob)
metrics.log_loss(y_test, y_prob) #.690 means that 


#Let's do this on the test set
testdf = pd.read_csv('test.csv', index_col=0)
testdf.head()

#We have to fit to all of the training data
logreg.fit(X, y)

#And then make our prediction on all of the test data
test_pred = logreg.predict_proba(testdf[feat_cols])[:,1]

#Creating the submission
sub = pd.DataFrame({'id':testdf.index, 'OpenStatus':test_pred}).set_index('id')
sub.to_csv('sub1.csv')




'''ATTEMPT #2'''
'''Creating Function To Modify All Files'''
def MakeFeatures(filename):
    df = pd.read_csv(filename, index_col=0)
    
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime': 'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    df['BodyLength'] = df.BodyMarkdown.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
    
    return df
'''Creating Function To Modify All Files'''


traindf = MakeFeatures('train.csv')
testdf = MakeFeatures('test.csv')


feat_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
X = traindf[feat_cols]
y = traindf.OpenStatus
logreg.fit(X, y)
sub2 = logreg.predict_proba(testdf[feat_cols])[:,1]
'''ATTEMPT #2'''








'''SERIES STRING METHODS IN PANDAS'''

#Is the entire title in lower case?
df.Title
df['TitleLowercase'] = (df.Title.str.lower() == df.Title).astype(int)
df.groupby(df.TitleLowercase).OpenStatus.mean()
#Is the entire title in lower case?

#Creating Text Features
df.Title.str.contains('need', case=False).sum()
#Creating Text Features

'''SERIES STRING METHODS IN PANDAS'''




'''USING COUNTVECTORIZER'''

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm
'''USING COUNTVECTORIZER'''



'''OTHER IDEAS'''


'''OTHER IDEAS'''





