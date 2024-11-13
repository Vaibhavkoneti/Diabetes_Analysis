import pandas as pd
import mlflow.sklearn

# Load the trained model from MLflow (replace <RUN_ID> with your actual run ID)
model = mlflow.sklearn.load_model("runs:/4c5735f857d140c8a240ddf7da7304f6/model")

# Example input for prediction (replace with actual patient data)
new_data = [[5, 116, 74, 0, 0, 35.5, 0.627, 26]]  # Example values

# Convert new_data to a DataFrame if necessary (depends on your model input shape)
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
new_data_df = pd.DataFrame(new_data, columns=columns)

# Make predictions
prediction = model.predict(new_data_df)
print("Prediction (1 indicates diabetes):", prediction[0])