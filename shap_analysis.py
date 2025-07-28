#!/usr/bin/env python3
"""
SHAP Model Interpretability Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

def generate_shap_analysis():
    """Generate SHAP values for model interpretability"""
    print("GENERATING SHAP ANALYSIS")
    print("-" * 40)
    
    # Load data and model
    data_path = "data/processed_features.csv"
    model_path = "models/lightgbm_model.pkl"
    
    if not Path(data_path).exists() or not Path(model_path).exists():
        print("Missing data or model files")
        return False
    
    # Load data
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in [
        'days_to_pending', 'RegionName', 'Date', 'StateName', 'SizeRank', 'RegionID', 'RegionType'
    ]]
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Load model
    model = joblib.load(model_path)
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (sample for performance)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # Create visualizations directory
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # SHAP Summary Plot
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Feature Importance - Housing Market Prediction')
    plt.tight_layout()
    plt.savefig(viz_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP Bar Plot
    print("Creating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Bar Chart)')
    plt.tight_layout()
    plt.savefig(viz_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance summary
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:25s}: {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv(viz_dir / 'shap_feature_importance.csv', index=False)
    
    print(f"\nSHAP analysis complete! Check {viz_dir}/ for visualizations")
    return True

if __name__ == "__main__":
    generate_shap_analysis()

