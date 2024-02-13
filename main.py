import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("encoded_data.csv")

# Assuming 'X' is your feature matrix (without 'item_no') and 'y' is the target variable
X = df.drop(['success_indicator'], axis=1)  # Assuming 'success_indicator' is the target variable
y = df['success_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines for each model
ann_pipeline = Pipeline([
    ('ann', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000))
])

rf_pipeline = Pipeline([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

svm_pipeline = Pipeline([
    ('svm', SVC(kernel='rbf', random_state=42))
])

# List of pipelines
pipelines = [ann_pipeline, rf_pipeline, svm_pipeline]

# Loop through pipelines and fit each model
for pipeline in pipelines:
    model = pipeline.named_steps[pipeline.steps[-1][0]]
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Model: {type(model).__name__}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    print(f"Classification Report:\n{classification_rep}\n")
