import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath='../data/hr_data.csv'):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}")
        return None

def clean_data(df):
    """
    Perform basic data cleaning.
    """
    # Example: Drop rows with missing values
    df_cleaned = df.dropna()
    
    # Example: Convert Attrition to binary (0/1) for modeling
    if 'Attrition' in df_cleaned.columns:
        df_cleaned['Attrition'] = df_cleaned['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df_cleaned

def preprocess_data(df):
    """
    Full preprocessing pipeline including encoding.
    """
    df = clean_data(df)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

def split_data(df, target_column='Attrition', test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the module
    df = load_data()
    if df is not None:
        print(df.head())
        df_processed = preprocess_data(df)
        print("Processed Data Head:")
        print(df_processed.head())
