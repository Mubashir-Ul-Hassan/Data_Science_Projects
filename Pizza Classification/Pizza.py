# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = '/path/to/pizzas.csv'  # Use the correct path for your file
data = pd.read_csv(file_path)

# Encode categorical columns (pizza_id, pizza_type_id, size) into numerical values
label_encoder = LabelEncoder()
data['pizza_id'] = label_encoder.fit_transform(data['pizza_id'])
data['pizza_type_id'] = label_encoder.fit_transform(data['pizza_type_id'])
data['size'] = label_encoder.fit_transform(data['size'])

# Separate features and target
X = data.drop('pizza_type_id', axis=1)  # Features (pizza_id, size, price)
y = data['pizza_type_id']               # Target (pizza_type_id)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(report)
