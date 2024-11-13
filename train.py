import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('data/diabetes.csv')

# Display basic information about the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Description:")
print(data.describe())

# Preprocess data (example: replace zero values in specific columns)
data['Glucose'].replace(0, data['Glucose'].mean(), inplace=True)
data['BloodPressure'].replace(0, data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].replace(0, data['SkinThickness'].mean(), inplace=True)
data['Insulin'].replace(0, data['Insulin'].mean(), inplace=True)
data['BMI'].replace(0, data['BMI'].mean(), inplace=True)

# Split data into features and target variable
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
mlflow.set_experiment("diabetes-prediction")
with mlflow.start_run():
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Print classification report for detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))