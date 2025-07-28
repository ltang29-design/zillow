#!/usr/bin/env python3
"""
Uncertainty Quantification for Housing Predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def bootstrap_uncertainty(model, X_train, y_train, X_test, n_bootstrap=100):
    """Calculate bootstrap uncertainty estimates"""
    predictions = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        
        # Train model on bootstrap sample
        model_boot = type(model)(**model.get_params())
        model_boot.fit(X_boot, y_boot)
        
        # Predict
        pred = model_boot.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    return {
        'mean': predictions.mean(axis=0),
        'std': predictions.std(axis=0),
        'lower_ci': np.percentile(predictions, 2.5, axis=0),
        'upper_ci': np.percentile(predictions, 97.5, axis=0)
    }

def generate_uncertainty_analysis():
    """Generate uncertainty quantification analysis"""
    print("GENERATING UNCERTAINTY ANALYSIS")
    print("-" * 40)
    
    # Load data
    data_path = "data/processed_features.csv"
    if not Path(data_path).exists():
        print("Missing data file")
        return False
    
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in [
        'days_to_pending', 'RegionName', 'Date', 'StateName', 'SizeRank', 'RegionID', 'RegionType'
    ]]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['days_to_pending']
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Load or train model
    model_path = "models/lightgbm_model.pkl"
    if Path(model_path).exists():
        print("Loading existing model...")
        model = joblib.load(model_path)
    else:
        print("Training RandomForest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # Generate bootstrap uncertainty
    print("Calculating bootstrap uncertainty (this may take a moment)...")
    uncertainty = bootstrap_uncertainty(model, X_train, y_train, X_test, n_bootstrap=50)
    
    # Create visualizations
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Uncertainty plot
    print("Creating uncertainty visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predictions with confidence intervals
    sort_idx = np.argsort(uncertainty['mean'])
    x_plot = np.arange(len(sort_idx))
    
    ax1.fill_between(x_plot, 
                     uncertainty['lower_ci'][sort_idx], 
                     uncertainty['upper_ci'][sort_idx],
                     alpha=0.3, label='95% Confidence Interval')
    ax1.plot(x_plot, uncertainty['mean'][sort_idx], 'b-', linewidth=2, label='Prediction')
    ax1.plot(x_plot, y_test.iloc[sort_idx].values, 'r.', alpha=0.5, label='Actual')
    ax1.set_xlabel('Sample Index (sorted by prediction)')
    ax1.set_ylabel('Days to Pending')
    ax1.set_title('Predictions with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty distribution
    ax2.hist(uncertainty['std'], bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Prediction Uncertainty (std)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate coverage rate
    actual_in_ci = ((y_test.values >= uncertainty['lower_ci']) & 
                    (y_test.values <= uncertainty['upper_ci']))
    coverage_rate = actual_in_ci.mean()
    
    print(f"\nUNCERTAINTY ANALYSIS RESULTS:")
    print("-" * 40)
    print(f"Coverage Rate: {coverage_rate:.3f} (target: 0.95)")
    print(f"Mean Uncertainty: {uncertainty['std'].mean():.3f} days")
    print(f"Median Uncertainty: {np.median(uncertainty['std']):.3f} days")
    
    # Save uncertainty results
    uncertainty_df = pd.DataFrame({
        'prediction': uncertainty['mean'],
        'uncertainty': uncertainty['std'],
        'lower_ci': uncertainty['lower_ci'],
        'upper_ci': uncertainty['upper_ci'],
        'actual': y_test.values
    })
    uncertainty_df.to_csv(viz_dir / 'uncertainty_results.csv', index=False)
    
    print(f"\nUncertainty analysis complete! Check {viz_dir}/ for results")
    return True

if __name__ == "__main__":
    generate_uncertainty_analysis() 