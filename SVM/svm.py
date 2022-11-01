import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\car_purchase.csv')
print(df)

print('\n\n')

#splitting dataset
x = df[['Age','AnnualSalary']]
print(x)

y = df['Purchased']
print('\n\n')
print(y)

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=42)
print()
print(x_train)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print('\n\n')
print(x_train)
print('\n\n')

#Fitting the SVM classifier

svc = SVC(kernel = 'rbf',random_state = 0)
svc.fit(x_train,y_train)

#predict
y_pred = svc.predict(x_test)
newdf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(newdf)

#Calculating confusion matrix and displaying classification report
print('\n\n')
print(confusion_matrix(y_test,y_pred))
print('\n\n')
print(classification_report(y_test,y_pred))