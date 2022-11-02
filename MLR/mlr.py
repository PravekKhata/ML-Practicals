import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

dataset = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\cars.csv')
print(dataset)

print(dataset.isnull().sum())
print('\n\n')
#Data analysis

print(dataset['transmission'].unique())
print(dataset['owner'].unique())
print('\n\n')

#data preprocessing

le = LabelEncoder()
dataset['transmission'] = le.fit_transform(dataset['transmission'])

oe = OrdinalEncoder(categories = [['Test Drive Car','First Owner','Second Owner','Third Owner','Fourth & Above Owner']],dtype = int)
dataset[['owner']] = oe.fit_transform(dataset[['owner']])
print('\n\n')
print(dataset)

x = dataset[['year','km_driven','transmission','owner']]
y = dataset['selling_price']

print('\n\n')
print(x)
print('\n\n')
print(y)
print('\n\n')

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=13)

print(x_train)
print('\n\n')

#MLR

x_train = x_train.values
print('X:')
print(x_train)
print()
print('Dimensions: ', x_train.shape)

print('\n\n')
y_train = y_train.values.reshape(len(y_train),1)
print('Y: ')
print(y_train)
print()
print("Dimesnsions: ",y_train.shape)

#Calculating theta

def calculate_theta(x,y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

theta = calculate_theta(x_train,y_train)
print('\n\n')
print('Parameters of theta: ')
print(theta)

#Predict

def predict(theta,x):
    return np.matmul(x,theta)

y_pred = predict(theta,x_test.values)
print('Predicted Value: ')
print('\n')
print(y_pred[0:10])

#comparison between actual and predict values
print('\n\n')
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(result)

