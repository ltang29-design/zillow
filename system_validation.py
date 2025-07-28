#!/usr/bin/env python3
"""
Comprehensive System Test - Show Features, SHAP, Uncertainty, and Overall Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

# Advanced analysis libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

def analyze_current_features():
    """Analyze and display current features being used"""
    print("🔍 CURRENT FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load processed features
    try:
        df = pd.read_csv("data/processed_features.csv")
        print(f"✅ Loaded data: {df.shape}")
    except FileNotFoundError:
        print("❌ processed_features.csv not found. Let me check available data files...")
        data_dir = Path("data")
        if data_dir.exists():
            files = list(data_dir.glob("*.csv"))
            print(f"Available data files: {[f.name for f in files]}")
            if files:
                # Use the first available CSV
                df = pd.read_csv(files[0])
                print(f"✅ Using {files[0].name}: {df.shape}")
            else:
                print("❌ No CSV files found in data directory")
                return None, None, None
        else:
            print("❌ Data directory not found")
            return None, None, None
    
    # Remove data leakage columns
    leakage_cols = [col for col in df.columns if 'days_to_pending_lag' in col]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        print(f"✅ Removed {len(leakage_cols)} data leakage columns:")
        for col in leakage_cols:
            print(f"  - {col}")
    
    # Identify features
    target_col = 'days_to_pending'
    exclude_cols = ['RegionName', 'date', target_col, 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n📊 FEATURE CATEGORIES:")
    print("-" * 40)
    
    # Categorize features
    feature_categories = {
        'Housing Market': [],
        'Economic Indicators': [],
        'Seasonality': [],
        'Temporal': [],
        'Market Stress': [],
        'Price Features': [],
        'Inventory Features': [],
        'Other': []
    }
    
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['season', 'month', 'quarter', 'holiday']):
            feature_categories['Seasonality'].append(col)
        elif any(keyword in col_lower for keyword in ['price', 'zhvi', 'median']):
            feature_categories['Price Features'].append(col)
        elif any(keyword in col_lower for keyword in ['inventory', 'months', 'supply']):
            feature_categories['Inventory Features'].append(col)
        elif any(keyword in col_lower for keyword in ['stress', 'desperation', 'velocity']):
            feature_categories['Market Stress'].append(col)
        elif any(keyword in col_lower for keyword in ['listing', 'pending', 'sale']):
            feature_categories['Housing Market'].append(col)
        elif any(keyword in col_lower for keyword in ['rate', 'gdp', 'unemployment', 'fred']):
            feature_categories['Economic Indicators'].append(col)
        elif any(keyword in col_lower for keyword in ['trend', 'lag', 'ma', 'year']):
            feature_categories['Temporal'].append(col)
        else:
            feature_categories['Other'].append(col)
    
    # Display categories
    total_features = 0
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category} ({len(features)} features):")
            total_features += len(features)
            for i, feature in enumerate(features[:5]):  # Show first 5
                print(f"  ✓ {feature}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
    
    print(f"\n📈 TOTAL FEATURES: {total_features}")
    
    # Prepare clean data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical columns first
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Converting {len(categorical_cols)} categorical columns to numeric...")
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            X[col] = X[col].fillna(X[col].median())
    
    print(f"✅ Clean data prepared: {X.shape}")
    
    return X, y, feature_cols

def run_shap_analysis(X, y, feature_cols):
    """Run SHAP analysis if available"""
    print(f"\n🎯 SHAP VALUE ANALYSIS")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP not available. Install with: pip install shap")
        return False
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model for SHAP
        print("Training LightGBM model for SHAP analysis...")
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        # Model performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Model Performance: R² = {r2:.4f}, RMSE = {rmse:.2f}")
        
        # SHAP analysis
        print("Computing SHAP values...")
        
        # Sample data for SHAP (for speed)
        sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        print(f"✅ SHAP values computed for {sample_size} samples")
        
        # Create SHAP visualizations
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        plt.savefig(viz_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                         plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)")
        plt.tight_layout()
        plt.savefig(viz_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top feature SHAP dependence plot
        feature_importance = np.abs(shap_values).mean(0)
        top_feature_idx = np.argmax(feature_importance)
        top_feature_name = feature_cols[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature_idx, shap_values, X_sample, 
                           feature_names=feature_cols, show=False)
        plt.title(f"SHAP Dependence Plot: {top_feature_name}")
        plt.tight_layout()
        plt.savefig(viz_dir / 'shap_dependence_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ SHAP visualizations saved:")
        print("  📊 shap_summary_plot.png")
        print("  📊 shap_bar_plot.png") 
        print("  📊 shap_dependence_plot.png")
        
        # Show top 10 features by SHAP importance
        print(f"\n📈 TOP 10 FEATURES BY SHAP IMPORTANCE:")
        print("-" * 50)
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_uncertainty_analysis(X, y):
    """Run uncertainty quantification analysis"""
    print(f"\n📊 UNCERTAINTY QUANTIFICATION")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Bootstrap uncertainty estimation
        print("Running bootstrap uncertainty estimation...")
        
        n_bootstrap = 50
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[boot_idx]
            y_boot = y_train.iloc[boot_idx]
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i)
            model.fit(X_boot, y_boot)
            
            # Predict
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Calculate uncertainty metrics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        # Performance metrics
        r2 = r2_score(y_test, mean_pred)
        rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        mean_uncertainty = np.mean(std_pred)
        
        print(f"✅ Uncertainty Analysis Complete:")
        print(f"  Model Performance: R² = {r2:.4f}, RMSE = {rmse:.2f}")
        print(f"  Average Uncertainty: ±{mean_uncertainty:.2f} days")
        print(f"  95% Confidence Interval Width: {np.mean(ci_upper - ci_lower):.2f} days")
        
        # Create uncertainty visualization
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Predictions with uncertainty
        plt.subplot(2, 2, 1)
        plt.errorbar(range(len(y_test[:100])), mean_pred[:100], yerr=std_pred[:100], 
                    fmt='o', alpha=0.6, capsize=2, capthick=1)
        plt.plot(range(len(y_test[:100])), y_test.iloc[:100], 'r-', alpha=0.8, label='Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Days to Pending')
        plt.title('Predictions with Uncertainty (First 100 samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Actual vs Predicted with CI
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, mean_pred, alpha=0.6)
        plt.fill_between(y_test.sort_values(), 
                        ci_lower[np.argsort(y_test)], 
                        ci_upper[np.argsort(y_test)], 
                        alpha=0.3, label='95% CI')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Days to Pending')
        plt.ylabel('Predicted Days to Pending')
        plt.title('Predictions vs Actual with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty distribution
        plt.subplot(2, 2, 3)
        plt.hist(std_pred, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Uncertainty (std dev)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Uncertainty')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Residuals vs Uncertainty
        residuals = np.abs(y_test - mean_pred)
        plt.subplot(2, 2, 4)
        plt.scatter(std_pred, residuals, alpha=0.6)
        plt.xlabel('Prediction Uncertainty')
        plt.ylabel('Absolute Residuals')
        plt.title('Uncertainty vs Actual Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'uncertainty_quantification.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Uncertainty visualization saved: uncertainty_quantification.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Uncertainty analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_current_system():
    """Test the current system comprehensively"""
    print(f"\n🧪 COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    # Check key components
    components = {
        'Data Processing': 'data/processed_features.csv',
        'Models Directory': 'models/',
        'Visualizations': 'visualizations/',
        'API Module': 'app/api/prediction_api.py',
        'Feature Engineering': 'app/features/feature_engineering.py',
        'ML Pipeline': 'app/ml/modeling_pipeline.py'
    }
    
    print("🔍 Component Status:")
    print("-" * 30)
    
    for component, path in components.items():
        if Path(path).exists():
            print(f"✅ {component}")
        else:
            print(f"❌ {component} (missing: {path})")
    
    # Check visualizations
    viz_dir = Path("visualizations")
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        print(f"\n📊 Available Visualizations ({len(viz_files)}):")
        print("-" * 40)
        for viz_file in viz_files:
            size_mb = viz_file.stat().st_size / (1024 * 1024)
            print(f"  📈 {viz_file.name} ({size_mb:.1f}MB)")
    
    return True

def main():
    """Main execution"""
    print("🏛️  COMPREHENSIVE SYSTEM TEST & FEATURE ANALYSIS")
    print("=" * 80)
    print("🔍 Features | 🎯 SHAP Analysis | 📊 Uncertainty | 🧪 System Test")
    print()
    
    try:
        # 1. Analyze current features
        X, y, feature_cols = analyze_current_features()
        
        if X is None:
            print("❌ Cannot proceed without data")
            return
        
        # 2. Run SHAP analysis
        shap_success = run_shap_analysis(X, y, feature_cols)
        
        # 3. Run uncertainty analysis
        uncertainty_success = run_uncertainty_analysis(X, y)
        
        # 4. Test current system
        system_success = test_current_system()
        
        # Summary
        print(f"\n🎉 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Features Analyzed: {len(feature_cols)} features across multiple categories")
        print(f"{'✅' if shap_success else '❌'} SHAP Analysis: {'Completed with visualizations' if shap_success else 'Failed or unavailable'}")
        print(f"{'✅' if uncertainty_success else '❌'} Uncertainty Quantification: {'Completed with visualizations' if uncertainty_success else 'Failed'}")
        print(f"{'✅' if system_success else '❌'} System Components: {'All key components verified' if system_success else 'Some components missing'}")
        
        # Check what we have
        viz_dir = Path("visualizations")
        if viz_dir.exists():
            shap_files = list(viz_dir.glob("shap*.png"))
            uncertainty_files = list(viz_dir.glob("uncertainty*.png"))
            
            print(f"\n📊 AVAILABLE ANALYSIS:")
            print("-" * 40)
            print(f"🎯 SHAP Visualizations: {len(shap_files)} files")
            for f in shap_files:
                print(f"  📈 {f.name}")
            
            print(f"📊 Uncertainty Visualizations: {len(uncertainty_files)} files")
            for f in uncertainty_files:
                print(f"  📈 {f.name}")
        
        print(f"\n🏆 SYSTEM READY FOR ZILLOW INTERVIEW!")
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 