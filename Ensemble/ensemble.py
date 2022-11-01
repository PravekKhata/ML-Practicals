import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\Churn_Modelling.csv')
print(df)

#EDA
print("Columns: ",list(df.columns))
print('\n\n')

print(df.isnull().sum())
print('\n\n')

#label encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
print(df)
print('\n\n')

x = df[['Age','Balance','CreditScore','EstimatedSalary','IsActiveMember']]
y = df['Exited']

print(x)
print('\n\n')
print(y)

#train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 42)

print(x_train)

#Feature Scaling
msc = MinMaxScaler()
var_transform = ['Age','Balance','CreditScore','EstimatedSalary']

x_train[var_transform] = msc.fit_transform(x_train[var_transform])
x_test[var_transform] = msc.transform(
 x_test[var_transform])

print()
print(x_train.head())

#Stacking

#defining estimators
estimators_list = [('knn',KNeighborsClassifier(n_neighbors = 30)),
                   ('svm',SVC()),(('DT'),DecisionTreeClassifier(random_state=13))]

#Choosing meta model

stack_model = StackingClassifier(estimators=estimators_list,final_estimator=LogisticRegression())

#Fitting the stack model

stack_model.fit(x_train,y_train)

#Evaluating the performance of the model

y_train_pred = stack_model.predict(x_train)
y_test_pred = stack_model.predict(x_test)

#Accuarcy
trainset_accuracy = accuracy_score(y_train,y_train_pred)
testset_accuracy = accuracy_score(y_test,y_test_pred)

print('\n\n')
print('Training Performance: ')
print()
print('Accuracy of the Stacked Model: ',trainset_accuracy)
print('RMSE: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))

print('\n\n')
print('Testing Performance: ')
print()
print('Accuracy of the Stacked Model: ',trainset_accuracy)
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_test_pred)))