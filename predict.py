'''
Created on Aug 25, 2013

@author: mlesko
'''

from __future__ import division
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

pd.set_printoptions(max_columns=100)

boolean_to_binary = lambda x: 1 if x else 0

def nominal_to_binaries(nominals, df, column, missing="Missing", other="Other", comparison="contains"):
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
        if comparison == "contains":
            df[column + nominal] = df[column].str.contains(nominal, na=False)
        else:
            df[column + nominal] = df[column] == nominal
        df[column + nominal] = df[column + nominal].map(boolean_to_binary)
        df[column + other] = df[column + other] + df[column + nominal]
        
    df[column + other] = df[column + other].map(lambda x: 1 if x == 0 else 0)
    del df[column]    

def featurize(df):
    nominal_to_binaries(nominals=['1', '2', '3'], df=df, column='Pclass')
    nominal_to_binaries(nominals=['A', 'B', 'C', 'D', 'E', 'F', 'G'], df=df, column='Cabin')
    nominal_to_binaries(nominals=['S', 'C', 'Q'], df=df, column='Embarked')
    nominal_to_binaries(nominals=['female', 'male'], df=df, column="Sex", comparison="equals")
    
    # how high_freq_titles was calculated
    '''
    print(df['Title'].value_counts())
    high_freq_title = df['Title'].value_counts() > 7
    df['Title'] = [index if high_freq_title[index] else "Other" for index in df['Title']]
    '''
    
    # set age for NaN to average by title
    
    high_freq_titles = ['Mr.', 'Miss.', 'Mrs.', 'Master.']
    df['Title'] = [pieces[1].split(" ")[1] for pieces in df['Name'].str.split(",")]
    del df['Name']
    df['TitleTemp'] = df['Title'].map(lambda x: x if x in high_freq_titles else "Other")
    age_by_title = df.groupby('TitleTemp')['Age'].mean()
    df['Age'] = np.where(df['Age'].isnull(), age_by_title[df['TitleTemp']], df['Age'])    
    del df['TitleTemp']
    nominal_to_binaries(nominals=high_freq_titles, df=df, column='Title')
    del df['Age']
    
    df['IsAlone'] = df['SibSp'] + df['Parch'] == 0
    df['IsAlone'] = df['IsAlone'].map(boolean_to_binary)
    del df['SibSp']
    del df['Parch']
    
    #print(df['Ticket'].str.upper().str.replace("[^A-Za-z]*", "").value_counts())
    del df['Ticket']
    del df['Fare']
    
    return df

def cross_validate(clf, df, features, target):
    scores = cross_validation.cross_val_score(clf, df[features], df[target], cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #clf.fit(df[features], df[target])
    #print(clf.feature_importances_)
    #print(features)
    
def test_cross_validations(df):
    features = df.columns[2:]
    classifiers = [
        {'label': 'Decision Tree', 'algorithm': tree.DecisionTreeClassifier()},
        {'label': 'Gradient Boost', 'algorithm': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)},
        {'label': 'Random Forest', 'algorithm': RandomForestClassifier(n_jobs=-1)},
        {'label': 'Gaussian Naive Bayes', 'algorithm': GaussianNB()},
        {'label': 'AdaBoost', 'algorithm': AdaBoostClassifier(n_estimators=100)},
        {'label': 'Linear SVC', 'algorithm': LinearSVC()}
    ]
    
    for classifier in classifiers: 
        print('\n' + classifier['label'])
        cross_validate(classifier['algorithm'], df, features, 'Survived')        
        print('\n' + classifier['label'] + ' Pipelined')
        pipeline = Pipeline([
          ('first', tree.DecisionTreeClassifier()),
          #('second', RandomForestClassifier(n_jobs=-1)),
          ('last', classifier['algorithm'])
        ])
        cross_validate(pipeline, df, features, 'Survived')

def predict(train_df, test_df):
    pipeline = Pipeline([
     ('first', tree.DecisionTreeClassifier()),
      ('last', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
    ])
    pipeline.fit(train_df[train_df.columns[2:]], train_df['Survived'])
    return pipeline.predict(test_df[test_df.columns[1:]])

def main():
    train_df = pd.read_csv("./train.csv")
    train_df = featurize(train_df)
    #test_cross_validations(train_df)
    
    test_df = pd.read_csv("./test.csv")
    test_df = featurize(test_df)
    print(test_df.head())
    predictions = pd.DataFrame(test_df['PassengerId'])
    predictions['Survived'] = pd.Series(predict(train_df, test_df))
    predictions.to_csv('./predictions.csv', index=False)
    

if __name__ == "__main__":
    main()

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
