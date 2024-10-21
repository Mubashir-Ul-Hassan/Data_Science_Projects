import pandas as pd

data=pd.read_csv('/content/Telecom Customers Churn.csv')

data.head()

data.shape

data.columns

data.drop('OnlineSecurity',inplace=True,axis=1)
data.drop('OnlineBackup',inplace=True,axis=1)
data.drop('DeviceProtection',inplace=True,axis=1)
data.drop('TechSupport',inplace=True,axis=1)
data.drop('StreamingTV',inplace=True,axis=1)
data.drop('StreamingMovies',inplace=True,axis=1)
data.drop('Contract',inplace=True,axis=1)
data.drop('PaperlessBilling',inplace=True,axis=1)
data.drop('PaymentMethod',inplace=True,axis=1)

data.drop('customerID',inplace=True,axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['Partner'] = label_encoder.fit_transform(data['Partner'])
data['Dependents'] = label_encoder.fit_transform(data['Dependents'])
data['PhoneService'] = label_encoder.fit_transform(data['PhoneService'])
data['MultipleLines'] = label_encoder.fit_transform(data['MultipleLines'])
data['InternetService'] = label_encoder.fit_transform(data['InternetService'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])

data.head()

import numpy as np
data['MonthlyCharges'] = data['MonthlyCharges'].replace(' ', np.nan)
data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = data['TotalCharges'].astype(float)

data.shape

X=data.drop('Churn',axis=1)
y=data['Churn']

X.shape

y.shape

