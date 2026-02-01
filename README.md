# Stock Market Prediction using Machine Learning

## Overview
This project implements a machine learning-based system to predict daily stock
price movements using historical market data and technical indicators.

The work was completed as an academic project and focuses on building a clear,
interpretable pipeline rather than complex black-box models.

## Problem Statement
Stock price movements are influenced by multiple dynamic factors, making prediction
challenging. This project aims to classify whether a stock’s price will increase
or decrease on the next trading day using historical data.

## Data
Historical stock data for five major technology companies was collected from
Yahoo Finance for the period January 2020 to December 2024:
- Apple (AAPL)
- Microsoft (MSFT)
- Alphabet (GOOGL)
- Amazon (AMZN)
- Meta (META)

## Methodology
- Computed technical indicators: RSI, 50-day SMA, 20-day EMA
- Used a Random Forest Classifier for prediction
- Trained separate models for each stock
- Evaluated performance using backtesting

## Results
The model achieved an average accuracy of approximately 60% across all stocks.
Microsoft (MSFT) achieved the highest accuracy at around 67%, with consistent
performance across different market conditions.

## Repository Structure
- `data_collection.py` – stock data retrieval
- `feature_engineering.py` – technical indicator computation
- `train_model.py` – model training
- `backtest.py` – strategy evaluation
- `visualize.ipynb` – result visualization
- `models/` – trained model files
- `stock_data/` – raw historical data
- `stock_features/` – processed feature data

## Status
Completed (Academic Project – May 2025)
