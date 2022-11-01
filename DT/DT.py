import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('D:\Desktop\ML\ML Pycharm Pracs\CollegeData.csv')
print(df)

print()
print("Columns: ", list(df.columns))
print()

print(df.isnull().sum())
print()

print(df.describe())
print()

#label encoding
le = LabelEncoder()

df['parent_was_in_college'] = le.fit_transform(df['parent_was_in_college'])
print()
print(df)

#############
x = df[['parent_age','average_grades']]
y = df['parent_was_in_college']

print('\n\n')
print(x)
print('\n\n')
print(y)
print()

#train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=1)

print('\n\n')
print(x_train)
print('\n\n')
print(y_train)
print('\n\n')

#Decision Tree Classifier (70-30 split)

dtclf = DecisionTreeClassifier(random_state = 13)

#Training the classifier
dtclf.fit(x_train,y_train)

#Predict
y_pred_train = dtclf.predict(x_train)
y_pred_test = dtclf.predict(x_test)
print('\n')

#Classification Report
print('Classification Report on Training Set: \n')
print(classification_report(y_train,y_pred_train))
print('\n\n')

print('Classification Report on Test Set: \n')
print(classification_report(y_test,y_pred_test))
print()



