from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("encode_data.csv")

# Assuming 'X' is your feature matrix (without 'item_no') and 'y' is the target variable
X = df.drop(['success_indicator'], axis=1)  # Assuming 'success_indicator' is the target variable
y = df['success_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 3: Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_report_svm = classification_report(y_test, y_pred_svm)

print("SVM Model Evaluation:")
print(f"Accuracy: {accuracy_svm}")
print(f"Classification Report:\n{classification_report_svm}\n")