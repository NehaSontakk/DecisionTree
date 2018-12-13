import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
#Read our csv data
data = pd.read_csv('/home/ubuntu/ml2.csv',names=['ID','age','income','gender','marital status','buys'])
print(data.head())
print(data.info())

#Identify target variable 
#Store the variable in string format
#Convert the string categorical values into an integer code using factorize method
data['buys'],buy_data = pd.factorize(data['buys'])
print(buy_data)
print(data['buys'].unique())

#Encode predictor variables 
data['ID'],_ = pd.factorize(data['ID'])
data['age'],_ = pd.factorize(data['age'])
data['income'],_ = pd.factorize(data['income'])
data['gender'],_ = pd.factorize(data['gender'])
data['marital status'],_ = pd.factorize(data['marital status'])
print(data.head())
print(data.info())

#Target and predictor variable features
X = data.iloc[:,:-1]
print("Our predictor features",X)
y = data.iloc[:,-1]
print("Our prediction class",y)

# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#Uses CART and GINI index
# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

#Predictor variables
print("X values for prediction",X_test)
print("Predicted value",y_pred)

#Checking incorrectly classified samples
incorr = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(incorr))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

import graphviz
features = X.columns

T = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=features,  
                                class_names=buy_data)
decision_tree = graphviz.Source(T)  
decision_tree.view()

