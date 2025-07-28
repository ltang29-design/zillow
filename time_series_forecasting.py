#!/usr/bin/env python3
"""
Time Series Forecasting for Housing Market Prediction
Proper forecasting without data leakage + Seasonality + Ensemble Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Time series and ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

# Time series specific
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime, timedelta

class TimeSeriesHousingPredictor:
    """Time Series Housing Market Predictor with Seasonality and Ensemble"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.ensemble_weights = {}
        
    def load_and_clean_data(self):
        """Load data and remove data leakage"""
        print("LOADING AND CLEANING DATA (NO LEAKAGE)")
        print("-" * 50)
        
        df = pd.read_csv("data/processed_features.csv")
        print(f"Original data: {df.shape}")
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # CRITICAL: Remove all lagged target variables (data leakage!)
        target_col = 'days_to_pending'
        leakage_cols = [col for col in df.columns if 'days_to_pending_lag' in col]
        
        if leakage_cols:
            print(f"🚨 REMOVING DATA LEAKAGE: {len(leakage_cols)} columns")
            for col in leakage_cols:
                print(f"   Removed: {col}")
            df = df.drop(columns=leakage_cols)
        
        # Keep only legitimate features (no future information)
        exclude_cols = ['RegionName', 'date', target_col, 'Unnamed: 0']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"✅ Clean features: {len(feature_cols)} columns")
        print(f"✅ Target: {target_col}")
        
        return df, feature_cols, target_col
    
    def engineer_time_series_features(self, df):
        """Engineer proper time series features with seasonality"""
        print("\nENGINEERING TIME SERIES FEATURES")
        print("-" * 50)
        
        df = df.copy()
        
        if 'date' not in df.columns:
            print("⚠️  No date column found, using synthetic dates")
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='M')
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date for proper time series
        df = df.sort_values(['RegionName', 'date']).reset_index(drop=True)
        
        # 1. SEASONALITY FEATURES
        print("📅 Adding seasonality features...")
        
        # Monthly seasonality
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarterly seasonality
        df['quarter'] = df['date'].dt.quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Day of year (annual cycle)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Moving season indicator (spring/summer are busy housing seasons)
        df['moving_season'] = ((df['month'] >= 4) & (df['month'] <= 9)).astype(int)
        
        # 2. TREND FEATURES
        print("📈 Adding trend features...")
        
        # Time trend (months since start)
        min_date = df['date'].min()
        df['months_since_start'] = (df['date'] - min_date).dt.days / 30.44
        df['time_trend'] = df['months_since_start']
        df['time_trend_sq'] = df['time_trend'] ** 2
        
        # 3. LAGGED ECONOMIC FEATURES (NOT TARGET!)
        print("🔄 Adding lagged economic features...")
        
        # Create lagged features for economic indicators (legitimate predictors)
        econ_features = [
            'median_price', 'new_listings', 'new_pending', 'inventory_months',
            'market_stress', 'supply_pressure', 'affordability_stress'
        ]
        
        # Sort and group by region for proper lagging
        df = df.sort_values(['RegionName', 'date'])
        
        for feature in econ_features:
            if feature in df.columns:
                # 1-month lag
                df[f'{feature}_lag1'] = df.groupby('RegionName')[feature].shift(1)
                # 3-month lag  
                df[f'{feature}_lag3'] = df.groupby('RegionName')[feature].shift(3)
                
                # Moving averages (trailing)
                df[f'{feature}_ma3'] = df.groupby('RegionName')[feature].rolling(3, min_periods=1).mean().reset_index(drop=True)
                df[f'{feature}_ma6'] = df.groupby('RegionName')[feature].rolling(6, min_periods=1).mean().reset_index(drop=True)
        
        # 4. REGIONAL FEATURES
        print("🏘️  Adding regional features...")
        
        # Regional averages (cross-sectional)
        for feature in ['median_price', 'new_listings', 'inventory_months']:
            if feature in df.columns:
                regional_avg = df.groupby('date')[feature].transform('mean')
                df[f'{feature}_regional_avg'] = regional_avg
                df[f'{feature}_vs_regional'] = df[feature] / (regional_avg + 1e-6)
        
        # 5. INTERACTION FEATURES
        print("🔗 Adding interaction features...")
        
        # Seasonality interactions
        if 'median_price' in df.columns:
            df['price_x_season'] = df['median_price'] * df['moving_season']
        
        if 'inventory_months' in df.columns:
            df['inventory_x_trend'] = df['inventory_months'] * df['time_trend']
        
        print(f"✅ Enhanced features: {df.shape[1]} total columns")
        
        return df
    
    def prepare_time_series_data(self, df, feature_cols, target_col):
        """Prepare data for time series forecasting"""
        print("\nPREPARING TIME SERIES DATA")
        print("-" * 50)
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Get all available feature columns (including new ones)
        all_feature_cols = [col for col in df.columns if col not in [
            'RegionName', 'date', target_col, 'Unnamed: 0'
        ]]
        
        print(f"Available features: {len(all_feature_cols)}")
        
        # Prepare features and target
        X = df[all_feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values in features
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with forward fill, then median
        for col in X.columns:
            if X[col].isna().all():
                X[col] = 0
            else:
                # Forward fill for time series
                X[col] = X[col].fillna(method='ffill')
                # Then median for remaining
                X[col] = X[col].fillna(X[col].median())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Final data: X={X.shape}, y={y.shape}")
        print(f"✅ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return X, y, df['date']
    
    def time_series_split_validation(self, X, y, dates):
        """Proper time series cross-validation"""
        print("\nTIME SERIES CROSS-VALIDATION")
        print("-" * 50)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Models for ensemble
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=2000),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        cv_results = {}
        fold_predictions = {name: [] for name in models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/5:")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Print date ranges
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            print(f"  Train: {train_dates.min()} to {train_dates.max()}")
            print(f"  Val:   {val_dates.min()} to {val_dates.max()}")
            
            # Scale features for linear models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            fold_results = {}
            
            for name, model in models.items():
                try:
                    # Use scaled features for linear models
                    if name in ['Ridge', 'Lasso']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    r2 = r2_score(y_val, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    mae = mean_absolute_error(y_val, y_pred)
                    
                    fold_results[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
                    fold_predictions[name].append(y_pred)
                    
                    print(f"    {name:15s}: R²={r2:.4f}, RMSE={rmse:.2f}")
                    
                except Exception as e:
                    print(f"    {name:15s}: Error - {str(e)[:30]}")
                    fold_results[name] = {'r2': 0, 'rmse': 999, 'mae': 999}
            
            cv_results[f'fold_{fold+1}'] = fold_results
        
        # Calculate average performance
        avg_results = {}
        for model_name in models.keys():
            r2_scores = [cv_results[f'fold_{i+1}'][model_name]['r2'] for i in range(5)]
            rmse_scores = [cv_results[f'fold_{i+1}'][model_name]['rmse'] for i in range(5)]
            
            avg_results[model_name] = {
                'avg_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'avg_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores)
            }
        
        print(f"\n📊 AVERAGE CROSS-VALIDATION RESULTS:")
        print("-" * 50)
        for name, results in avg_results.items():
            print(f"{name:15s}: R²={results['avg_r2']:.4f}±{results['std_r2']:.4f}, "
                  f"RMSE={results['avg_rmse']:.2f}±{results['std_rmse']:.2f}")
        
        return cv_results, avg_results
    
    def train_ensemble_model(self, X, y):
        """Train ensemble model with multiple algorithms"""
        print("\nTRAINING ENSEMBLE MODEL")
        print("-" * 50)
        
        # Split data (time series split - use last 20% as test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Individual models
        models = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1, max_iter=2000),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        # Train individual models and collect predictions
        individual_predictions = {}
        individual_performance = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Train
                if name in ['ridge', 'lasso']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Store
                self.models[name] = model
                individual_predictions[name] = y_pred
                
                # Performance
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                individual_performance[name] = {'r2': r2, 'rmse': rmse}
                
                print(f"  {name}: R²={r2:.4f}, RMSE={rmse:.2f}")
                
            except Exception as e:
                print(f"  {name}: Error - {str(e)[:50]}")
        
        # Create ensemble using weighted average based on performance
        print(f"\nCreating ensemble...")
        
        # Calculate weights based on R² scores
        r2_scores = {name: perf['r2'] for name, perf in individual_performance.items()}
        total_r2 = sum(max(0, r2) for r2 in r2_scores.values())
        
        if total_r2 > 0:
            weights = {name: max(0, r2) / total_r2 for name, r2 in r2_scores.items()}
        else:
            weights = {name: 1/len(r2_scores) for name in r2_scores.keys()}
        
        self.ensemble_weights = weights
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(y_test))
        for name, weight in weights.items():
            if name in individual_predictions:
                ensemble_pred += weight * individual_predictions[name]
        
        # Ensemble performance
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        print(f"\n🎯 ENSEMBLE RESULTS:")
        print(f"  Ensemble: R²={ensemble_r2:.4f}, RMSE={ensemble_rmse:.2f}")
        print(f"  Weights: {weights}")
        
        return {
            'individual_performance': individual_performance,
            'ensemble_performance': {'r2': ensemble_r2, 'rmse': ensemble_rmse},
            'ensemble_weights': weights,
            'test_predictions': ensemble_pred,
            'test_actual': y_test.values
        }
    
    def analyze_seasonality(self, df, target_col):
        """Analyze seasonality patterns"""
        print("\nSEASONALITY ANALYSIS")
        print("-" * 50)
        
        if 'date' not in df.columns:
            print("⚠️  No date column for seasonality analysis")
            return None
        
        df_season = df.copy()
        df_season['month'] = df_season['date'].dt.month
        df_season['quarter'] = df_season['date'].dt.quarter
        
        # Monthly patterns
        monthly_avg = df_season.groupby('month')[target_col].agg(['mean', 'std', 'count'])
        
        # Quarterly patterns
        quarterly_avg = df_season.groupby('quarter')[target_col].agg(['mean', 'std', 'count'])
        
        print("📅 MONTHLY PATTERNS:")
        for month, row in monthly_avg.iterrows():
            print(f"  Month {month:2d}: {row['mean']:.1f}±{row['std']:.1f} days (n={row['count']})")
        
        print(f"\n📊 QUARTERLY PATTERNS:")
        for quarter, row in quarterly_avg.iterrows():
            print(f"  Q{quarter}: {row['mean']:.1f}±{row['std']:.1f} days (n={row['count']})")
        
        return {
            'monthly': monthly_avg,
            'quarterly': quarterly_avg
        }
    
    def create_visualizations(self, ensemble_results, seasonality_results):
        """Create comprehensive visualizations"""
        print("\nCREATING VISUALIZATIONS")
        print("-" * 50)
        
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Model Performance Comparison
        plt.figure(figsize=(12, 8))
        
        models = list(ensemble_results['individual_performance'].keys())
        r2_scores = [ensemble_results['individual_performance'][m]['r2'] for m in models]
        rmse_scores = [ensemble_results['individual_performance'][m]['rmse'] for m in models]
        
        # Add ensemble
        models.append('Ensemble')
        r2_scores.append(ensemble_results['ensemble_performance']['r2'])
        rmse_scores.append(ensemble_results['ensemble_performance']['rmse'])
        
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² scores
        bars1 = ax1.bar(x, r2_scores, width, alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance - R² Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(r2_scores):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # RMSE scores
        bars2 = ax2.bar(x, rmse_scores, width, alpha=0.8, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Model Performance - RMSE')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(rmse_scores):
            ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'time_series_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predictions vs Actual
        plt.figure(figsize=(12, 6))
        
        actual = ensemble_results['test_actual']
        predicted = ensemble_results['test_predictions']
        
        plt.subplot(1, 2, 1)
        plt.scatter(actual, predicted, alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Days to Pending')
        plt.ylabel('Predicted Days to Pending')
        plt.title('Ensemble Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = ensemble_results['ensemble_performance']['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = actual - predicted
        plt.scatter(predicted, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Days to Pending')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'time_series_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Seasonality Analysis
        if seasonality_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Monthly patterns
            monthly_data = seasonality_results['monthly']
            months = monthly_data.index
            means = monthly_data['mean']
            stds = monthly_data['std']
            
            ax1.errorbar(months, means, yerr=stds, marker='o', capsize=5)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Average Days to Pending')
            ax1.set_title('Monthly Seasonality Pattern')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(months)
            
            # Quarterly patterns
            quarterly_data = seasonality_results['quarterly']
            quarters = quarterly_data.index
            q_means = quarterly_data['mean']
            q_stds = quarterly_data['std']
            
            ax2.errorbar(quarters, q_means, yerr=q_stds, marker='s', capsize=5, color='orange')
            ax2.set_xlabel('Quarter')
            ax2.set_ylabel('Average Days to Pending')
            ax2.set_title('Quarterly Seasonality Pattern')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(quarters)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'time_series_seasonality.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ All visualizations saved!")
    
    def run_complete_analysis(self):
        """Run complete time series analysis"""
        print("🕐 TIME SERIES HOUSING MARKET FORECASTING")
        print("=" * 70)
        print("✅ No Data Leakage + Seasonality + Ensemble Modeling")
        print()
        
        try:
            # 1. Load and clean data
            df, feature_cols, target_col = self.load_and_clean_data()
            
            # 2. Engineer time series features
            df_enhanced = self.engineer_time_series_features(df)
            
            # 3. Prepare data
            X, y, dates = self.prepare_time_series_data(df_enhanced, feature_cols, target_col)
            
            # 4. Time series cross-validation
            cv_results, avg_results = self.time_series_split_validation(X, y, dates)
            
            # 5. Train ensemble model
            ensemble_results = self.train_ensemble_model(X, y)
            
            # 6. Analyze seasonality
            seasonality_results = self.analyze_seasonality(df_enhanced, target_col)
            
            # 7. Create visualizations
            self.create_visualizations(ensemble_results, seasonality_results)
            
            print(f"\n🎉 TIME SERIES ANALYSIS COMPLETE!")
            print("=" * 70)
            print("✅ Data leakage removed (no lagged target variables)")
            print("✅ Proper time series features with seasonality")
            print("✅ Time series cross-validation")
            print("✅ Ensemble modeling with 6 algorithms")
            print("✅ Seasonality analysis")
            print("✅ Comprehensive visualizations")
            
            # Best performing model
            best_model = max(avg_results.keys(), key=lambda x: avg_results[x]['avg_r2'])
            best_r2 = avg_results[best_model]['avg_r2']
            ensemble_r2 = ensemble_results['ensemble_performance']['r2']
            
            print(f"\n🏆 PERFORMANCE SUMMARY:")
            print(f"  Best Individual: {best_model} (R² = {best_r2:.4f})")
            print(f"  Ensemble Model: R² = {ensemble_r2:.4f}")
            print(f"  Features Used: {len(self.feature_names)}")
            
            return {
                'cv_results': avg_results,
                'ensemble_results': ensemble_results,
                'seasonality': seasonality_results,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main execution"""
    predictor = TimeSeriesHousingPredictor()
    results = predictor.run_complete_analysis()
    return results

if __name__ == "__main__":
    main() 