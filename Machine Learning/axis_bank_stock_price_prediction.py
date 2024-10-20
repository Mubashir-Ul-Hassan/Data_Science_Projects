import numpy as np 
import pandas as pd 
from sklearn import metrics 
import matplotlib.pyplot as plt 
%matplotlib inline 
from sklearn.model_selection import train_test_split

stock=pd.read_csv('/kaggle/input/axis-bank/AXISBANK.csv')

stock.head()

stock.shape

stock['Date']=pd.to_datetime(stock.Date)

List_drop=['%Deliverble','Deliverable Volume']
stock.drop(List_drop,axis=1,inplace=True)

#Visualize the Open price data
plt.figure(figsize=(16,8))
plt.title('AXISBANK')
plt.xlabel('Days')
plt.ylabel('Close price IND (&)')
plt.plot(stock['Open'])
plt.show()

stock.info()

stock.describe()

stock.isnull()

stock.isnull().sum()

print(len(stock))

stock['Open'].plot(figsize=(12,8))

stock['High'].plot(figsize=(12,8))

stock['Low'].plot(figsize=(12,8))

stock['Last'].plot(figsize=(12,9))

stock['Close'].plot(figsize=(12,9))

stock['VWAP'].plot(figsize=(12,9))

stock['Volume'].plot(figsize=(12,9))

stock.hist(figsize=(12,8),bins=50)

# here we createed a dependent and independent variables
X=stock[['Open','High','Low','Volume']]
y=stock['Close']

#Spliting data into train data and test data 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
