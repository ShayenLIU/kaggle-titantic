'''
Created on Aug 25, 2013

@author: mlesko
'''

from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

boolean_to_binary = lambda x: 1 if x else 0

def nominal_to_binaries(nominals, df, column, missing="Missing", other="Other"):
    ''' Convert nominal into binary column per nominal + a "Missing" and "Other" column 
    kwargs
    nominals   -- list of nominal values expected
    df         -- pandas dataframe
    column     -- name of column in dataframe
    na         -- value to assign to column name
    other      -- value to assign to column name    
    '''
    
    df[column] = df[column].astype('string') # this converts NaN to string "nan"
    
    df[column + missing] = df[column].map(lambda x: x == "nan")    
    df[column + missing] = df[column + missing].map(boolean_to_binary)
    df[column + other] = df[column + missing]

    for nominal in nominals:        
        df[column + nominal] = df[column].str.contains(nominal, na=False)
        df[column + nominal] = df[column + nominal].map(boolean_to_binary)
        df[column + other] = df[column + other] + df[column + nominal]
        
    df[column + other] = df[column + other].map(lambda x: 1 if x == 0 else 0)
    #del df[column]    

df = pd.read_csv("./train.csv")

nominal_to_binaries(nominals=['1', '2', '3'], df=df, column='Pclass')
nominal_to_binaries(nominals=['A', 'B', 'C', 'D', 'E', 'F', 'G'], df=df, column='Cabin')
nominal_to_binaries(nominals=['S', 'C', 'Q'], df=df, column='Embarked')

# how high_freq_titles was calculated
'''
print(df['Title'].value_counts())
high_freq_title = df['Title'].value_counts() > 7
df['Title'] = [index if high_freq_title[index] else "Other" for index in df['Title']]
'''

high_freq_titles = ['Mr.', 'Miss.', 'Mrs.', 'Master.']
df['Title'] = [pieces[1].split(" ")[1] for pieces in df['Name'].str.split(",")]
del df['Name']
df['Title'] = df['Title'].map(lambda x: x if x in high_freq_titles else "Other")
age_by_title = df.groupby('Title')['Age'].mean()
    
nominal_to_binaries(nominals=high_freq_titles, df=df, column='Title')

df['Age'].fillna(df['Age'].mean())

df['IsAlone'] = df['SibSp'] + df['Parch'] == 0
df['IsAlone'] = df['IsAlone'].map(boolean_to_binary)
del df['SibSp']
del df['Parch']


pd.set_printoptions(max_columns=100)
print(df.head(40))





#print(df['Ticket'].str.upper().str.replace("[^A-Za-z]*", "").value_counts())
del df['Ticket']

# factorize columns
df['Sex'] = pd.factorize(df['Sex'])[0]
#df['Embarked'] = pd.factorize(df['Embarked'])[0]
df['Title'] = pd.factorize(df['Title'])[0]

#print(df.columns)

'''
features = df.columns[2:]
rfc = RandomForestClassifier(n_jobs=2)
scores = cross_validation.cross_val_score(rfc, df[features], df['Survived'], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''



'''
sklearn Random Forest & Pandas
http://blog.yhathq.com/posts/random-forests-in-python.html

Cross Validation
http://scikit-learn.org/stable/modules/cross_validation.html

Feature Selection
http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

Links on Competion
http://gertlowitz.blogspot.com.au/2013/06/where-am-i-up-to-with-titanic-competion.html
http://www.kaggle.com/c/titanic-gettingStarted/forums/t/4707/easy-or-difficult

Rudi Kruger Cabin Data 
http://www.kaggle.com/c/titanic-gettingStarted/forums/t/4693/is-cabin-an-important-predictor/26629#post26629

Categorical Data & sklearn
http://www.kaggle.com/c/titanic-gettingStarted/forums/t/5379/handling-categorical-data-with-sklearn

Categorical Data 
http://www.kaggle.com/c/titanic-gettingStarted/forums/t/4069/top-20-kagglers-i-am-wondering-what-they-are-doing/22732#post22732
'''
