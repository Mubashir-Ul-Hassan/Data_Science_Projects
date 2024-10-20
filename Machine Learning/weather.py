import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error , r2_score , root_mean_squared_error 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("D:\\seattle-weather.csv")
data

x=data[["precipitation",'temp_max','temp_min','wind']]
y=data["weather"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=3000)
