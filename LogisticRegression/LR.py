import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\car_purchase.csv')
print(dataset)

independent_features = ['Age','AnnualSalary']
x = dataset[independent_features]
print(x)
print()

y = dataset['Purchased']
print(y)
print()

#train test split

x_train, x_test, y_train,\
y_test = train_test_split(x,y,test_size=0.25,random_state=16)
print()
print(x_train)

#fitting logistic model

modellogistic = LogisticRegression( )
modellogistic.fit(x_train, y_train)

y_pred = modellogistic.predict(x_test)

#confusion matrix
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)
print()
print(cm)

#Classification report

cr = classification_report(y_test,y_pred)
print()
print(cr)

#Accuracy score
print()
print('Accuracy Score:')
print(accuracy_score(y_test,y_pred))