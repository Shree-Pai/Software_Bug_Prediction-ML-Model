import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
dataset_path = 'C:/Users/SHREELAKSHMI PAI/Downloads/bug_prediction_dataset_500.csv'  # Replace with your dataset path
try:
    data = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the path.")

# Step 2: Data Preprocessing
# Handle missing values
data.ffill(inplace=True)

# Check if 'Severity' column exists
if 'Severity' not in data.columns:
    raise ValueError("The dataset must contain a 'Severity' column.")

# Encode categorical variables (if any)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature-target split
X = data.drop('Severity', axis=1)  # Replace 'Severity' with your target column name
y = data['Severity']

# Encode target if necessary
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# 1. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 3. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# Step 4: Evaluation
models = {'SVM': svm_pred, 'Random Forest': rf_pred, 'KNN': knn_pred}
accuracies = {}

print("Classification Reports:\n")
for model_name, predictions in models.items():
    print(f"{model_name}:")
    print(classification_report(y_test, predictions))
    accuracies[model_name] = accuracy_score(y_test, predictions)

# Plot Confusion Matrices
cm_labels = np.unique(y_test)  # Dynamic labels
plt.figure(figsize=(15, 5))
for i, (model_name, predictions) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test, predictions, labels=cm_labels)
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.show()
