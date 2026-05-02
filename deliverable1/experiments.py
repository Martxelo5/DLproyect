import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import random

def train_and_evaluate(model, train_loader, val_loader, test_tensors, scaler_y, name="Model", epochs=100, seed=42):
    """Trains a single model and evaluates it on the unseen test set."""
    set_seed(seed)
    print(f"Training {name} for {epochs} epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    # Training Loop
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation on Test Set
    model.eval()
    x_test_tensor, y_test_tensor = test_tensors
    with torch.no_grad():
        scaled_preds = model(x_test_tensor).numpy()
        actual_preds = scaler_y.inverse_transform(scaled_preds)
        actual_y = scaler_y.inverse_transform(y_test_tensor.numpy())
        mae_dollars = np.mean(np.abs(actual_preds - actual_y))

    print(f"{name} Final Test MAE: ${mae_dollars:.2f}")
    return mae_dollars

def run_bias_variance_experiment(models_dict, train_loader, val_loader, test_tensors, scaler_y, epochs=100, seed=42):
    """Trains three different model architectures side-by-side and plots their learning curves."""
    set_seed(seed)
    results = {}
    criterion = nn.MSELoss()

    for name, model in models_dict.items():
        print(f"Training {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            iter_train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                iter_train_loss += loss.item()
            train_losses.append(iter_train_loss / len(train_loader))
            
            # Validate
            model.eval()
            iter_val_loss = 0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_targets)
                    iter_val_loss += v_loss.item()
            val_losses.append(iter_val_loss / len(val_loader))
            
        # Test Evaluation
        model.eval()
        x_test_tensor, y_test_tensor = test_tensors
        with torch.no_grad():
            scaled_preds = model(x_test_tensor).numpy()
            actual_preds = scaler_y.inverse_transform(scaled_preds)
            actual_y = scaler_y.inverse_transform(y_test_tensor.numpy())
            mae_dollars = np.mean(np.abs(actual_preds - actual_y))
            
        results[name] = {'train_loss': train_losses, 'val_loss': val_losses, 'test_mae': mae_dollars}

    # Plot the results
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Bias-Variance Tradeoff: Learning Curves Comparison', fontsize=16)

    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res['train_loss'], label='Train Loss', color='blue')
        ax.plot(res['val_loss'], label='Val Loss', color='orange')
        ax.set_title(f"{name}\nTest Error: ${res['test_mae']:.2f}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig('deliverable1/plots/bias_variance_curves.png')
    plt.close()
    print("- Saved Bias-Variance plot to plots/bias_variance_curves.png")

    # Print Final Summary
    print("\n--- BIAS-VARIANCE FINAL RESULTS ---")
    for name, res in results.items():
        print(f"{name}: ${res['test_mae']:.2f}")

def set_seed(seed=42):
    """Sets all seeds to ensure total reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_optuna_search(train_loader, val_loader, test_tensors, scaler_y, baseline_mae, n_trials=30, seed=42):
    """Dynamically searches for the best architecture, then tests the winner against the baseline."""
    
    # 1. Set global seed before starting the search
    set_seed(seed)

    def objective(trial):
        # 2. CRITICAL: Reset seed at the start of each trial 
        # so the model architecture always starts with the same initial weights
        set_seed(seed)
        
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layers = []
        in_features = 9 # Remember this is 9 based on your EDA/Data Prep (e.g., One-Hot regions)
        
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 16, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features 
            
        layers.append(nn.Linear(in_features, 1))
        model = nn.Sequential(*layers)
        
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(50): # Shorter epochs just for the search phase
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_targets)
                val_loss += v_loss.item()
                
        return val_loss / len(val_loader)

    # 3. Setup Optuna Study with a seeded Sampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    
    print(f"Running Optuna search with {n_trials} trials (Seed: {seed})...")
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # 4. Show the winning parameters
    print("\n--- OPTUNA WINNING ARCHITECTURE ---")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # 5. Build the Champion Model using the winning parameters
    # Reset seed again to ensure the final training is reproducible
    set_seed(seed)
    
    print("\nTraining the 'Challenger' Model (Optuna Winner) for 100 epochs...")
    layers = []
    in_features = 9
    for i in range(best_params['n_layers']):
        out_features = best_params[f'n_units_l{i}']
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    
    optimized_model = nn.Sequential(*layers)
    optimizer = optim.Adam(optimized_model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    # 6. Train the Challenger
    for epoch in range(100):
        optimized_model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = optimized_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 7. Evaluate on the Test Set
    optimized_model.eval()
    x_test_tensor, y_test_tensor = test_tensors
    with torch.no_grad():
        scaled_preds = optimized_model(x_test_tensor).numpy()
        actual_preds = scaler_y.inverse_transform(scaled_preds)
        actual_y = scaler_y.inverse_transform(y_test_tensor.numpy())
        optuna_mae = np.mean(np.abs(actual_preds - actual_y))
 
    # 8. We create the real vs predicted visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_y, actual_preds, alpha=0.5, color='purple', label='Optuna Predictions')

    # We draw the perfect prediction line (diagonal)
    max_val = int(max(actual_y.max(), actual_preds.max()))
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Real Charges ($)')
    plt.ylabel('Predicted Charges ($)')
    plt.title('Optuna Champion Model: Reality vs. Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('deliverable1/plots/optuna_scatter_plot.png')
    #plt.show()
    print("- Optuna scatter plot saved")

    # 9. The Final Showdown
    print("\n==============================================")
    print("COMPARISON")
    print("==============================================")
    print(f"Baseline Model MAE: ${baseline_mae:.2f}")
    print(f"Optuna Model MAE:   ${optuna_mae:.2f}")
    
    if optuna_mae < baseline_mae:
        improvement = baseline_mae - optuna_mae
        print(f"\nResult: Optuna wins! It improved the error by ${improvement:.2f}.")
    else:
        print("\nResult: Baseline wins!")
    
    return optimized_model