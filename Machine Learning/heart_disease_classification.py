import pandas as pd

data=pd.read_csv('/content/heart2.csv')

data.head()

data.shape

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['ChestPainType'] = label_encoder.fit_transform(data['ChestPainType'])
data['RestingECG'] = label_encoder.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = label_encoder.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = label_encoder.fit_transform(data['ST_Slope'])

data.head()

X=data.drop('HeartDisease',axis=1)
y=data['HeartDisease']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=15)

model.fit(X_train,y_train)

print('Accuracy Score:',model.score(X_test,y_test))
