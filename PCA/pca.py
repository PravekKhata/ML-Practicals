import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing the csv file
df = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\Bank.csv')
print(df)

print('\n\n')

#assigning features to x and y
x = df[['variance','skewness','curtosis']]
print(x)
print('\n\n')
y = df['class']
print(y)
print('\n\n')

#train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print('\n\n')
print(y_train)

print('\n\n')

#Feature scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print('\n\n')

#PCA

pca = PCA(n_components=1)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print('\n\n')
print(x_train)

#Fitting logistic regression to train set

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

#Predicting on test set
y_pred = classifier.predict(x_test)
print()
print(y_pred)

#finding accuracy
print()
print('Accuracy: ',accuracy_score(y_test,y_pred))



