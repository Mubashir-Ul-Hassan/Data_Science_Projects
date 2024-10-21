import pandas as pd
import numpy as np

data=pd.read_csv('/content/data.csv')

data.head()

data.shape

data.isnull().sum()

from sklearn.model_selection import train_test_split

x=data.drop('TAX',axis=1)
y=data['TAX']

x.shape

y.shape

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

