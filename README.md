# Housing Market Prediction System

A comprehensive machine learning system for predicting "days to pending" in US metropolitan housing markets using advanced econometric techniques and ensemble modeling.

## 🎯 Project Overview

This system predicts the time it takes for houses to go from listing to pending sale across 586 US metropolitan areas, achieving **65.1% R²** performance with **±9.93 days RMSE** using an Enhanced Ensemble approach.

## 🏗️ Architecture

```
Data Sources → Feature Engineering → Machine Learning → API → Dashboard
     ↓               ↓                    ↓           ↓        ↓
  Zillow CSV    46 Features        Enhanced      FastAPI   Interactive
  FRED API      Engineered         Ensemble      Endpoints    Charts
  Census API    (No Leakage)       65.1% R²      Real-time   Predictions
  Weather       Clean, Validated   ±9.93 RMSE    Cached      Uncertainty
```

## 📊 Key Features

### Data Sources
- **Zillow Research Data**: Housing metrics, pricing, inventory
- **Federal Reserve (FRED)**: Economic indicators, interest rates
- **US Census Bureau**: Demographics, income data
- **Weather Data**: Climate patterns affecting market seasonality

### Feature Engineering (46 Features)
- **Economic Indicators**: Mortgage rates, unemployment, GDP
- **Seasonality Features**: Monthly, quarterly, holiday patterns
- **Market Stress Indices**: Velocity, desperation, market friction
- **Price Features**: Momentum, volatility, appreciation rates
- **Housing Fundamentals**: Supply, demand, inventory levels

### Machine Learning
- **Enhanced Ensemble**: LightGBM + XGBoost + RandomForest
- **Time Series Cross-Validation**: Prevents data leakage
- **Uncertainty Quantification**: Bootstrap confidence intervals
- **Model Interpretability**: SHAP values for feature importance

### API & Deployment
- **FastAPI**: Real-time predictions (<200ms response)
- **PostgreSQL**: Normalized data storage
- **Redis**: Caching for performance
- **Interactive Dashboard**: HTML/JavaScript visualization

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PostgreSQL
Redis (optional)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/housing-market-prediction.git
cd housing-market-prediction

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env.template .env
# Edit .env with your API keys and database settings

# Initialize system
python setup.py
```

### Running the System
```bash
# Start the complete system
python quick_start.py

# Or run components individually:
python cli.py init-db              # Initialize database
python cli.py fetch-external-data  # Get FRED/Census data
python cli.py ingest-data          # Process Zillow data
```

### API Usage
```bash
# Start API server
uvicorn app.main:app --reload

# Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer housing-api-token-2025" \
  -H "Content-Type: application/json" \
  -d '{"metro_name": "Austin", "date": "2025-08-01"}'
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **R² Score** | 65.1% |
| **RMSE** | ±9.93 days |
| **Metro Coverage** | 586 US areas |
| **API Response Time** | <200ms |
| **Uncertainty** | ±2.31 days average |

## 🏛️ Technical Stack

- **Python**: Pandas, Scikit-learn, LightGBM, XGBoost
- **API**: FastAPI, PostgreSQL, Redis
- **Analysis**: SHAP, Bootstrap Uncertainty, Time Series CV
- **Visualization**: Matplotlib, Interactive HTML Dashboard

## 📁 Project Structure

```
housing-market-prediction/
├── app/
│   ├── core/                   # Configuration and database
│   ├── models/                 # SQLAlchemy ORM models
│   ├── features/               # Feature engineering pipeline
│   ├── ml/                     # Machine learning models
│   ├── api/                    # FastAPI endpoints
│   ├── external_data/          # Data source clients
│   ├── data_ingestion/         # Data transformation
│   └── dashboard/              # Interactive dashboard
├── analysis/
│   ├── time_series_forecasting.py
│   ├── uncertainty_analysis.py
│   ├── cv_regularization_analysis.py
│   └── shap_analysis.py
├── visualizations/             # Generated charts and plots
├── requirements.txt
├── cli.py                      # Command line interface
└── README.md
```

## 🎯 Model Insights

### Market Efficiency Patterns
- **Fast Markets** (18-22 days): Austin, Denver, Phoenix
- **Moderate Markets** (23-30 days): Dallas, Houston, Seattle
- **Slow Markets** (30+ days): Chicago, Miami, Northeast metros

### Seasonal Patterns
- **Spring/Summer**: 15-25% faster than winter
- **Peak Season** (April-July): Shortest days to pending
- **Holiday Periods**: 20-30% slower transactions

### Economic Sensitivity
- **1% mortgage rate increase**: +3-5 days to pending
- **High unemployment areas**: +10-15 days longer
- **Price volatility**: Indicates uncertainty, slower decisions

## 🔧 Advanced Features

### Econometric Techniques
- **Panel Data Models**: Fixed/Random effects (when applicable)
- **Data Leakage Prevention**: Removed target-dependent features
- **Multicollinearity Detection**: VIF analysis and correlation cleanup
- **Feature Selection**: Mutual information and domain knowledge

### Uncertainty Quantification
- **Bootstrap Resampling**: 50 iterations for confidence intervals
- **Prediction Intervals**: 95% confidence bounds
- **Model Uncertainty**: Ensemble disagreement metrics

### Model Interpretability
- **SHAP Values**: Feature contribution analysis
- **Feature Importance**: Cross-validated rankings
- **Partial Dependence**: Individual feature effects

## 📊 Business Applications

### Sales Optimization
- **Market Timing**: Target fast markets for inventory turnover
- **Pricing Strategy**: Adjust pricing in slow markets
- **Lead Prioritization**: Focus on high-efficiency metros

### Marketing Intelligence
- **Geo-Targeting**: Optimize campaigns by market efficiency
- **Seasonal Planning**: Time campaigns with market patterns
- **Customer Expectations**: Set realistic timeline expectations

### Operations Planning
- **Resource Allocation**: Staff based on market efficiency
- **Process Optimization**: Streamline for market conditions
- **Performance Metrics**: Track actual vs predicted efficiency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/housing-market-prediction](https://github.com/yourusername/housing-market-prediction)

## 🙏 Acknowledgments

- Zillow Research for housing market data
- Federal Reserve Economic Data (FRED) for economic indicators
- US Census Bureau for demographic data
- Open source machine learning community 