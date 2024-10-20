import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error , r2_score , root_mean_squared_error 
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


df=sns.load_dataset("tips")
df

x=df[["total_bill"]]
y=df["tip"]

sns.scatterplot(x="total_bill",y="tip",data=df)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=200)

