from logger import logger
from market_data_fetcher import (
    get_sp500_companies,
    get_ticker_to_cik_map,
    get_ticker_historical_prices,
    extract_quarterly_data,
)

import pandas as pd


def get_PE_ratio_data(ticker:str, ticker_price_data: pd.DataFrame, company_facts: dict):
    eps_table = extract_quarterly_data(company_facts, "EarningsPerShareBasic", 'USD/shares')
    if eps_table.empty:
        logger.warning(f"No EPS data found for {ticker}")
        return pd.DataFrame()
    eps_table = eps_table.copy()
    # ensure datetime types
    
    eps_table["start"] = pd.to_datetime(eps_table["start"], errors="coerce")
    eps_table["end"] = pd.to_datetime(eps_table["end"], errors="coerce")
    for col in ("start", "end"):
        eps_table[col] = pd.to_datetime(eps_table[col], errors="coerce")

    # drop rows missing crucial data
    eps_table = eps_table.dropna(subset=["start", "end", "val"])
    if eps_table.empty:
        logger.warning(f"No valid EPS rows after cleaning for {ticker}")
        return pd.DataFrame()

    # for each end date, pick the row with the latest start date and extract its val
    idx = eps_table.groupby("end")["start"].idxmax()
    latest_per_end = eps_table.loc[idx].sort_values("end").reset_index(drop=True)

    eps_df = latest_per_end[["end", "val"]].rename(columns={"val": "eps"})
    logger.info(f"Extracted {len(eps_df)} EPS points for {ticker}")
    
    Stock_price = ticker_price_data.loc[eps_df["end"], "Close"].reset_index(drop=True)
    eps_df["PE_ratio"] = Stock_price / eps_df["eps"]
    eps_df["price"] = Stock_price
    
    return eps_df

def get_PB_ratio_data(ticker:str, ticker_price_data: pd.DataFrame, company_facts: dict):
    equity_table = extract_quarterly_data(company_facts, 'StockholdersEquity', 'USD')
    shares_df = extract_quarterly_data(company_facts, 'CommonStockSharesOutstanding', 'shares')