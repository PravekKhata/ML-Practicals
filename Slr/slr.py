import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\Tvmarketing.csv')
print(dataset)

x = dataset['TV']
y = dataset['Sales']

# Train test split
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
print('X train')
print()
print(x_train)

print()

# slr - analytical method

def linear_reg(x, y):
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    b1_num = np.sum(((x - x_mean) * (y - y_mean)))
    b1_den = np.sum((x - x_mean) ** 2)
    b1 = b1_num / b1_den

    b0 = y_mean - (b1 * x_mean)

    return (b0, b1)


def predict(x, b0, b1):
    return b0 + (b1 * x)


b0, b1 = linear_reg(x_train, y_train)

print('B0: ', b0)
print()
print('B1: ', b1)
print()

y_pred = predict(x_test,b0,b1)
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(result)

print('\n\n')
x_ip = float(input('Enter value: '))
print()
print('Predicted Value: ', predict(x_ip,b0,b1))
print()