# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm

```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values 
```

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M SRINATH
RegisterNumber: 212222230147


import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```

## Output:

Original data(first five columns): 

![output](g1.png)
Data after dropping unwanted columns(first five): 

![output](g2.png)
Checking the presence of null values:

![output](g3.png)

Checking the presence of duplicated values:

![output](g4.png)

Data after Encoding:
![output](g5.png)
X Data:

![output](g6.png)
Y Data:

![output](g7.png)


Predicted Values:

![output](g8.png)


Accuracy Score:

![output](g9.png)

Confusion Matrix:

![output](g10.png)


Classification Report:

![output](g11.png)


Predicting output from Regression Model

![output](g12.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
