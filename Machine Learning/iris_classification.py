import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
iris=load_iris()

iris.feature_names

iris.target_names

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df.shape

df['target']=iris.target
df.head()

df[df.target==1].head()

df[df.target==2].head()

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()

df[45:55]
df0=df[:50]
df1=df[50:100]
df2=df[100:]

import matplotlib.pyplot as plt
%matplotlib inline

plt.xlabel('Sepal length')
plt.ylabel("Sepal Width")
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')

from sklearn.model_selection import train_test_split

x=df.drop(['target','flower_name'],axis='columns')
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

len(x_train)

len(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

knn.score(x_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
cm

knn.predict(x_test)

%matplotlib inline 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(12,8))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

