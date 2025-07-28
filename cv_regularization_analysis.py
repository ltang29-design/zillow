#!/usr/bin/env python3
"""
Cross-Validation and Regularization Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

def comprehensive_cv_analysis():
    """Comprehensive cross-validation analysis"""
    print("COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("-" * 50)
    
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
    
    # Scale features for linear models
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models with different regularization
    models = {
        'Ridge (a=0.1)': Ridge(alpha=0.1),
        'Ridge (a=1.0)': Ridge(alpha=1.0),
        'Ridge (a=10.0)': Ridge(alpha=10.0),
        'Lasso (a=0.1)': Lasso(alpha=0.1),
        'Lasso (a=1.0)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_results = {}
    
    for model_name, model in models.items():
        try:
            print(f"Testing {model_name}...")
            
            # Use scaled features for linear models
            if 'Ridge' in model_name or 'Lasso' in model_name or 'Elastic' in model_name:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
            else:
                scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            
            cv_results[model_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
            print(f"   R2 = {scores.mean():.4f} +/- {scores.std():.4f}")
            
        except Exception as e:
            print(f"   Error: {str(e)[:50]}")
    
    # Create visualization
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # CV Results Plot
    model_names = list(cv_results.keys())
    means = [cv_results[name]['mean'] for name in model_names]
    stds = [cv_results[name]['std'] for name in model_names]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(model_names, means, xerr=stds, capsize=5, alpha=0.7)
    plt.xlabel('R2 Score')
    plt.title('Cross-Validation Results with Regularization')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(mean + std + 0.01, i, f'{mean:.3f}+/-{std:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'cv_regularization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCross-validation analysis complete!")
    return cv_results

def regularization_path_analysis():
    """Analyze regularization paths"""
    print("\nREGULARIZATION PATH ANALYSIS")
    print("-" * 50)
    
    # Load data
    data_path = "data/processed_features.csv"
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in [
        'days_to_pending', 'RegionName', 'Date', 'StateName', 'SizeRank', 'RegionID', 'RegionType'
    ]]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['days_to_pending']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Regularization parameters
    alphas = np.logspace(-3, 2, 50)
    
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ridge regularization path
    print("Analyzing Ridge regularization path...")
    ridge_scores = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
        ridge_scores.append(scores.mean())
    
    ax1.semilogx(alphas, ridge_scores, 'b-', linewidth=2)
    best_ridge_alpha = alphas[np.argmax(ridge_scores)]
    ax1.axvline(best_ridge_alpha, color='red', linestyle='--', 
                label=f'Best a = {best_ridge_alpha:.4f}')
    ax1.set_xlabel('Regularization Parameter (alpha)')
    ax1.set_ylabel('R2 Score')
    ax1.set_title('Ridge Regularization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lasso regularization path
    print("Analyzing Lasso regularization path...")
    lasso_scores = []
    for alpha in alphas:
        try:
            lasso = Lasso(alpha=alpha, max_iter=2000)
            scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring='r2')
            lasso_scores.append(scores.mean())
        except:
            lasso_scores.append(np.nan)
    
    ax2.semilogx(alphas, lasso_scores, 'g-', linewidth=2)
    valid_scores = [s for s in lasso_scores if not np.isnan(s)]
    if valid_scores:
        best_lasso_idx = np.nanargmax(lasso_scores)
        best_lasso_alpha = alphas[best_lasso_idx]
        ax2.axvline(best_lasso_alpha, color='red', linestyle='--',
                    label=f'Best a = {best_lasso_alpha:.4f}')
    ax2.set_xlabel('Regularization Parameter (alpha)')
    ax2.set_ylabel('R2 Score')
    ax2.set_title('Lasso Regularization Path')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'regularization_paths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Best Ridge alpha: {best_ridge_alpha:.4f}")
    if valid_scores:
        print(f"Best Lasso alpha: {best_lasso_alpha:.4f}")
    
    return {
        'ridge': {'best_alpha': best_ridge_alpha, 'scores': ridge_scores},
        'lasso': {'best_alpha': best_lasso_alpha if valid_scores else None, 'scores': lasso_scores}
    }

def main():
    """Main execution"""
    print("ADVANCED ML ANALYSIS: CV + REGULARIZATION")
    print("=" * 60)
    
    # Run analyses
    cv_results = comprehensive_cv_analysis()
    reg_results = regularization_path_analysis()
    
    print("\nANALYSIS COMPLETE!")
    print("Cross-validation with multiple models")
    print("Regularization path analysis")
    print("Visualizations saved to visualizations/")
    
    return cv_results, reg_results

if __name__ == "__main__":
    main() 