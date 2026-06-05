import joblib
import numpy as np
import pandas as pd
import os

def predict_iris(features):
    """
    Loads the trained SVM classifier, scales the input features, and makes a prediction.
    Features: sepal length, sepal width, petal length, petal width
    """
    model_path = 'level_2_intermediate/models/classifier_svm.pkl'
    scaler_path = 'level_2_intermediate/models/scaler.pkl'
    le_path = 'level_2_intermediate/models/label_encoder.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        print("Error: Model artifacts not found. Please run the Level 2 Classification notebook first.")
        return None

    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    # Prepare and scale features as a DataFrame to keep feature names
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    
    # Predict and decode
    prediction = model.predict(features_scaled)
    predicted_label = le.inverse_transform(prediction)[0]
    
    return predicted_label


if __name__ == "__main__":
    print("--- Iris Species Inference Tool ---")
    
    # Sample input: [sepal_length, sepal_width, petal_length, petal_width]
    # For a Versicolor flower
    sample_input = [5.9, 2.7, 4.2, 1.3]
    
    result = predict_iris(sample_input)
    if result:
        print(f"Input Features: {sample_input}")
        print(f"Predicted Species: {result}")

