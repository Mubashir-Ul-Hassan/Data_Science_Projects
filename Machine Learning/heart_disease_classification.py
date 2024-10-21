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

