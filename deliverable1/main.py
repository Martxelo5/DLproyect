import numpy as np
import torch
import random

# Import from your other local files
import data_prep as dp
import models as m
import experiments as exp
import EDA as eda

SEED = 24684

def set_seed(seed=42):
    # Locks all random number generators for perfect reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you happen to run this on a GPU:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("Starting Deep Learning Insurance Project...")
    print("="*50)
    print(f"Using SEED: {SEED}\n")

    set_seed(SEED)

    # We save the path to the insurance csv
    data_path = "data/insurance.csv"

    # 0.Exploratory Data Analysis (EDA)
    eda.run_eda(filepath=data_path)
 
    # 1.Data Preparation
    train_loader, val_loader, test_tensors, scaler_y, raw_test_data = dp.load_and_prep_data(filepath=data_path, random_state=SEED)
    
    # 2.Linear Regression Baseline
    dp.run_linear_regression_baseline(raw_test_data, scaler_y)
    
    # 3.Neural Network Baseline
    print("\n---2: Neural Network Baseline ---")
    baseline_model = m.get_Baseline_model()
    # We train and evaluate the modeles in the 100 epoch loop and print the MAE
    baseline_mae = exp.train_and_evaluate(baseline_model, train_loader, val_loader, test_tensors, scaler_y, name="Baseline NN", seed=SEED)
    
    # 4.Bias-Variance Experiment
    print("\n---3: Bias-Variance Tradeoff Experiment ---")
    tiny_model = m.get_tiny_model()
    massive_model = m.get_massive_model()

    # We pass all three models to the experiment function to compare them
    exp.run_bias_variance_experiment(
        models_dict={
            "Under-parameterized (Tiny)": tiny_model,
            "Baseline (Current)": m.get_Baseline_model(),
            "Over-parameterized (Massive)": massive_model
        },
        train_loader=train_loader,
        val_loader=val_loader,
        test_tensors=test_tensors,
        scaler_y=scaler_y,
        seed=SEED
    )
    
    # 5.Optuna Architecture Search
    print("\n---4: Dynamic Optuna Architecture Search ---")
    exp.run_optuna_search(train_loader, val_loader, test_tensors, scaler_y, baseline_mae, seed=SEED)
    
    print("\n==============================================")
    print("Project Execution Complete!")

if __name__ == "__main__":
    main()