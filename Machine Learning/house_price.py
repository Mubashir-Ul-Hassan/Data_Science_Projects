import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error , r2_score , root_mean_squared_error 
from sklearn import svm
from sklearn.metrics import accuracy_score


data=pd.read_csv("D:\\house_price_regression_dataset.csv")
data

x=data[["Square_Footage","Num_Bedrooms","Num_Bathrooms","Lot_Size","Garage_Size","Neighborhood_Quality"]]
y=data["House_Price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=3000)

model=RandomForestClassifier(n_estimators=100,random_state=10)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(accuracy_score(y_test,y_pred))
