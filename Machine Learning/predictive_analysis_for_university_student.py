import numpy as np 
import pandas as pd 
import os 
for dirname, _,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

df=pd.read_csv('/kaggle/input/undergraduate-admission-test-survey-in-bangladesh/Undergraduate Admission Test Survey in Bangladesh.csv')

df.head()

df.info()

df.shape

df.isnull().sum()

df.duplicated().sum()

df.describe()

# Calculate the mean of the HSC_GPAcolumn 
mean_hec_gpa=df['HSC_GPA'].mean()
#Replace null values with the mean
df['HSC_GPA']=df['HSC_GPA'].fillna(mean_hec_gpa)

df.drop_duplicates(inplace=True)

pip install autoviz

from autoviz.AutoViz_Class import AutoViz_Class
Av=AutoViz_Class()
%matplotlib inline
#you have specify the target variable 
dft=Av.AutoViz('/kaggle/input/undergraduate-admission-test-survey-in-bangladesh/Undergraduate Admission Test Survey in Bangladesh.csv', depVar = 'University')

#Define the feature and the target
X=df.drop('University',axis=1)
y=df['University']


