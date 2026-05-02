import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def load_and_prep_data(filepath="data/insurance.csv", batch_size=64, random_state=42):
    """
    Loads the insurance dataset, preprocesses it, and returns DataLoaders 
    along with the scalers and raw test data for evaluation.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # Manual coding of binary variables
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

    # One-Hot Encoding for the region variable (Yields 4 columns -> 9 features total)
    df = pd.get_dummies(df, columns=['region'])

    # Define input (X) and target (y) variables
    X = df.drop('charges', axis=1).values
    y = df['charges'].values.reshape(-1, 1)

    # 1 Partition: In train(70%) and a temporal variable(30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # 2.Partition: We divide the temporal variable(30%) to validation(15%) and test(15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Scale the data (Standardization: mean=0, variance=1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # Convert Numpy arrays to PyTorch Tensors
    x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    # Note: We usually don't need a test_loader since we evaluate the test set all at once

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # We package up the specific variables that main.py needs
    test_tensors = (x_test_tensor, y_test_tensor)
    raw_test_data = (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    return train_loader, val_loader, test_tensors, scaler_y, raw_test_data

def run_linear_regression_baseline(raw_test_data, scaler_y):
    """Runs a basic Linear Regression to establish a baseline error."""
    print("\n---1: Linear Regression Baseline ---")
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = raw_test_data
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    
    lr_scaled_preds = lr_model.predict(X_test_scaled)
    lr_actual_preds = scaler_y.inverse_transform(lr_scaled_preds)
    actual_y_test = scaler_y.inverse_transform(y_test_scaled)
    
    lr_mae = mean_absolute_error(actual_y_test, lr_actual_preds)
    print(f"Linear Regression Test MAE: ${lr_mae:.2f}") 