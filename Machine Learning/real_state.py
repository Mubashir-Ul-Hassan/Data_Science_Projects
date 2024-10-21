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

model=KNeighborsClassifier(n_neighbors=3)

X_train = X_train.dropna()  # Drop rows with NaNs
y_train = y_train[X_train.index]  # Align labels

model.fit(X_train,y_train)

from sklearn.impute import SimpleImputer

# Create a SimpleImputer object
imputer = SimpleImputer(strategy='mean')  # Choose a strategy (mean, median, etc.)

# Fit the imputer to your training data (assuming you have it)
imputer.fit(X_train)

# Now you can use the imputer to transform your test data
X_test_imputed = imputer.transform(X_test)

# Ensure X_test_imputed has the same number of samples as y_test
if X_test_imputed.shape[0] != y_test.shape[0]:
    raise ValueError("X_test_imputed and y_test have different shapes after imputation.")

print('Accuracy Score:', model.score(X_test_imputed, y_test))
