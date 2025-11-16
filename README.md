# Stock Correlation Prediction

## Overview
This project analyzes correlations among NASDAQ S&P 500 stocks using graph-based machine learning techniques. By constructing a correlation graph where nodes represent stocks and edges represent statistical relationships, we identify communities of related stocks, detect systemic patterns, and predict next-window correlations

## Key Features
- **Market Data Retrieval**: Fetches historical stock prices and financial data from Yahoo Finance and SEC filings
- **Feature Engineering**: Computes technical indicators (momentum, price changes, volatility) and fundamental metrics (P/E ratio, P/B ratio, ROA, current ratio)
- **Graph Construction**: Builds dynamic correlation networks with configurable window sizes and correlation thresholds
- **Correlation Prediction**: Predicts next-window stock correlations using both linear and non-linear models
- **Flexible Configuration**: Adjustable hyperparameters (window size, correlation threshold, time periods)

## Project Structure

### `src/` Directory
- **`company_feature_functions.py`**: Computes financial and technical features for stocks
  - P/E ratio, P/B ratio, ROA, current ratio extraction
  - Historical price-based metrics (momentum, percentage changes, volatility)
  
- **`graph_builder.py`**: Constructs and manages correlation graphs
  - `GraphManager` class for building dynamic stock correlation networks
  - Configurable window sizes and correlation thresholds
  
- **`feature_lists.py`**: Defines all feature categories
  - Historical data features, PE/PB/ROA features, current ratio features
  
- **`market_data_fetcher/`**: Handles external data retrieval
  - `financial_api_functions.py`: Fetches S&P 500 company list, ticker-to-CIK mapping, SEC data
  - `helper_functions.py`: Utilities for processing SEC quarterly data
  
- **`logger.py`**: Centralized logging configuration
  
- **`feature_engineering.md`**: Documentation of feature definitions

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd COMP4222_F25-26
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment (if needed)**
   - Create a `.env` file if external API keys are required

## Running the Project

To train the model:
- run the train.py file in `/src`


## Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `yfinance`: Historical stock price data
- `requests`: HTTP requests for data fetching
- `ta-lib`: Technical analysis indicators
- `torch`, `torch-geometric`: Graph neural network support (optional for future extensions)

## Project Details
- **Data Period**: 2020-2025
- **Stock Universe**: S&P 500 companies
- **Correlation Window**: 5-day rolling windows (configurable)
- **Correlation Threshold**: 0.8 (configurable)
- **Models**: TwoTowerSAGE

---

**COMP4222 Fall 2025 Course Project**  
By Harsh, Lakshya, and Monish