import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv("D:\\asiacup.csv")

# Define features and target
x = data[['Toss','Selection','Run_Scored','Wicket_Lost','Fours','Sixes','Extras','Run_Rate','Avg_Bat_Strike_Rate','Highest_Score','Wicket_Taken']]
y = data['Result']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=15)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=15)

# Train the model
rf_classifier.fit(x_train, y_train)

# Predict on test data
y_pred = rf_classifier.predict(x_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print(f'Accuracy Score: {accuracy:.2f}')