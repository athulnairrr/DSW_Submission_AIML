from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.rad_csv("encoded_data.csv")

# Assuming 'X' is your feature matrix (without 'item_no') and 'y' is the target variable
X = df.drop(['success_indicator'], axis=1)  # Assuming 'success_indicator' is the target variable
y = df['success_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Artificial Neural Network (ANN)
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)
classification_report_ann = classification_report(y_test, y_pred_ann)

print("ANN Model Evaluation:")
print(f"Accuracy: {accuracy_ann}")
print(f"Classification Report:\n{classification_report_ann}\n")