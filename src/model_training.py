from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import os
from data_preprocessing import load_data, preprocess_data, split_data

# Ensure data path is correct relative to execution context
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/hr_data.csv')

def train_model(model_type='logistic_regression'):
    """
    Train a machine learning model on the HR dataset.
    """
    df = load_data(DATA_PATH)
    if df is None:
        return None
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df_processed, target_column='Attrition')
    
    # Initialize model
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    # Train model
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    model_filename = f'../models/{model_type}_model.joblib'
    if not os.path.exists('../models'):
        os.makedirs('../models')
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    return model

if __name__ == "__main__":
    train_model('random_forest')
