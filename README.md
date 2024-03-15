# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SABARI S
RegisterNumber: 212222240085

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
Placement Data:
![image](https://user-images.githubusercontent.com/119389139/233679600-d7637871-ac7e-4ef8-8538-cfe8f8c1ddb3.png)

Salary Data:
![image](https://user-images.githubusercontent.com/119389139/233679823-32ae13cc-489d-436a-925a-6187d6de27ed.png)

Checking the null() function:











![image](https://user-images.githubusercontent.com/119389139/233679969-7a2b5524-270d-4377-9728-f78188177f6c.png)

Data Duplicate:



![image](https://user-images.githubusercontent.com/119389139/233680057-efb79829-4a73-4fab-9e37-01f58b54898b.png)

Print Data:
![image](https://user-images.githubusercontent.com/119389139/233680198-69570dd4-1cce-4363-bce2-a4cae93e236e.png)

Data-status:




![image](https://user-images.githubusercontent.com/119389139/233680590-861937d3-aba8-400c-8ccf-80c25444cd69.png)

y_prediction array:





![image](https://user-images.githubusercontent.com/119389139/233680712-229c768c-f1c1-4ec8-b43f-0b0d2996ee31.png)

Accuracy value:





![image](https://user-images.githubusercontent.com/119389139/233680788-7cbdbe90-d08b-4076-aac7-50a4ad6c26b0.png)

Confusion array:




![image](https://user-images.githubusercontent.com/119389139/233681147-aca68fa8-33ae-48e3-b8db-bfe884d619ee.png)

Classification report:





![image](https://user-images.githubusercontent.com/119389139/233681332-f1ee5ca5-9812-40b9-8d7b-c3cda844fec3.png)

Prediction of LR:


![image](https://user-images.githubusercontent.com/119389139/233681412-e62e2859-e43f-4515-8a18-ae7ea8bc19cb.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
