import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report, f1_score


from xgboost import XGBClassifier


df= pd.read_csv("Churn_Modelling.csv")

#Check how many rows and columns we have 
#print(df.shape)

#print(df.info)

#checking for null values 
#print(df.isnull().sum())

#this gives the statistical summary of what is going on
#print(df.describe())

#The first thing is to do exploritory anaylis: Finding corrilation to one column with another column
#if cell<0 negatively correlated meaning there's less relationship on between that label and the feature
#if cell>0 positively correlated meaning there's likely to be a relatioship between a potential feature and a label. 
plt.figure(figsize=(15,10))
numeric_df = df.select_dtypes(include=[np.number])  #select only numeric values from the columns
corr = numeric_df.corr()
#print(corr)
#sns.heatmap(corr,annot=True, cmap="Accent")

#Skip the first part of the table
#Graph showing correlation between the values of the column
# sorted_correlation= corr['Exited'].sort_values(ascending= False)[1:]
# #sns.barplot(x=sorted_correlation.index, y=sorted_correlation.values)

# print(df['Exited'].value_counts())
# sns.countplot(x='Exited', data=df)

#print(sorted_correlation)


#if say you wanted to see which customers are male or female

# print(df['Gender'].value_counts())
# sns.countplot(x='Gender', data=df)
# plt.show()

#Now let's check which feature is important
#sns.countplot(x='Exited', data=df, hue='Gender',)
#sns.FacetGrid(df, col='Exited').map(sns.displot,"Age")


#Data cleaning
df['Geography']=df['Geography'].astype('category').cat.codes # this converts geographical areas into code['France, Germany', Spain]-> [0,1,2]
df.drop(columns=['RowNumber', "CustomerId", "Surname"], inplace=True) #remove all irrelevant information
df['Gender'] = np.where(df['Gender'] == 'Female', 1, 0)
 #if gender==male then 1 

X=df.drop(columns=['Exited']).values
y=df['Exited'].values

#print(X.shape)
#print(y.shape)

df.head()

#Now let's split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Let's create a model 
basic_model=XGBClassifier()
basic_model.fit(X_train,y_train)

print(basic_model)


def evaluate_model(model):
    print("Training accuracy:", model.score(X_train,y_train))
    print("Test accuracy:", model.score(X_test,y_test))
    y_pred=model.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("F1 score: ", f1_score(y_test,y_pred))
evaluate_model(basic_model)

#Now let's fine tune the model 


#So the  f1 Score is less than the test accuracy what does that mean??
# Fine tuning the model 
xgb_params={
    'n_estimators': [100, 200],
    'max_depth': [4,5]
}

grid= GridSearchCV(XGBClassifier(), param_grid= xgb_params, scoring= 'accuracy', cv=10)

grid.fit(X_train, y_train)

print("Best Score", grid.best_score_)

print( "Best params score", grid.best_estimator_)


model_1=XGBClassifier(max_depth=4)
model_1.fit(X_train, y_train)

evaluate_model(model_1)
#plt.show()
