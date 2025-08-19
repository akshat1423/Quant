# Algorithmic Trading & Option Pricing using ML & Monte Carlo

A comprehensive quantitative finance project implementing advanced option pricing models and machine learning-based trading strategies.

## Project Overview

This project implements:
- **Option Pricing**: American Put Options using Stochastic Mesh & Longstaff-Schwartz algorithms
- **Trading Strategies**: DON, Actor-Critic, and Epsilon-Greedy reinforcement learning models
- **Dashboard**: Streamlit-based data analysis interface
- **Cloud Integration**: Azure ML deployment and Azure App Services hosting

## Key Features

### Option Pricing Models
- American Put Options with Stochastic Mesh method
- Longstaff-Schwartz algorithm for early exercise optimization
- Binomial and European option models for comparison
- Mean Squared Error < 0.05 vs benchmark models

### Machine Learning Trading Strategies
- Deep Option Network (DON) model
- Actor-Critic reinforcement learning
- Epsilon-Greedy exploration strategy
- Azure ML integration for training and deployment

### Data Analysis Dashboard
- Interactive Streamlit dashboard
- Real-time data visualization
- Performance metrics and analysis
- Deployed on Azure App Services

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option Pricing
```python
from option_pricing import AmericanPutPricer, LongstaffSchwartz

# Price American put option
pricer = AmericanPutPricer()
price = pricer.price_stochastic_mesh(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
```

### Trading Strategies
```python
from trading_strategies import ActorCriticStrategy

# Initialize trading strategy
strategy = ActorCriticStrategy()
strategy.train(market_data)
```

### Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

## Project Structure

- `option_pricing/`: Option pricing models and Monte Carlo engines
- `trading_strategies/`: ML-based trading strategy implementations
- `data/`: Data loading and market data utilities
- `dashboard/`: Streamlit dashboard and visualizations
- `azure_integration/`: Azure ML pipeline and deployment scripts
- `tests/`: Unit tests and validation

## Performance Metrics

- Option pricing MSE < 0.05 compared to binomial/European models
- Trading strategy backtesting with Sharpe ratio optimization
- Real-time performance monitoring through dashboard

## Technologies Used

- Python, NumPy, SciPy, Pandas
- TensorFlow, PyTorch for deep learning
- Stable-Baselines3 for reinforcement learning
- QuantLib for financial modeling
- Streamlit for dashboard
- Azure ML and Azure App Services for cloud deployment