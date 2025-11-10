from .logger import logger
from .market_data_fetcher import (
    get_sp500_companies,
    get_ticker_to_cik_map,
    get_ticker_historical_prices,
    extract_quarterly_data,
)
from .helper_functions import clean_eps_table

import pandas as pd


def get_PE_ratio_data(
    ticker: str, ticker_price_data: pd.DataFrame, company_facts: dict, start_year: int, end_year: int
):
    eps_table = extract_quarterly_data(
        company_facts, "EarningsPerShareBasic", "USD/shares"
    )
    if eps_table.empty:
        logger.warning(f"No EPS data found for {ticker}")
        return pd.DataFrame()
    eps_table = eps_table.copy()
    # ensure datetime types

    eps_table["start"] = pd.to_datetime(eps_table["start"], errors="coerce")
    eps_table["end"] = pd.to_datetime(eps_table["end"], errors="coerce")

    eps_table = clean_eps_table(eps_table, start_year, end_year)
    logger.info(f"Extracted {len(eps_table)} EPS points for {ticker}")

    # Ensure price index is datetime and sorted
    price_df = ticker_price_data.copy()
    price_df.index = pd.to_datetime(price_df.index, errors="coerce")
    price_series = price_df["Close"].sort_index()

    # Align prices to the EPS 'end' dates by taking the last available close on or before each end date
    # using forward-fill on a reindex to the eps_table end dates.
    try:
        aligned_prices = price_series.reindex(eps_table["end"], method="ffill")
    except Exception:
        # fallback to simple reindex if method fails (e.g., incompatible indexes)
        aligned_prices = price_series.reindex(eps_table["end"])

    # Ensure aligned_prices is a Series (if DataFrame was returned, take the first column)
    if isinstance(aligned_prices, pd.DataFrame):
        if not aligned_prices.empty:
            aligned_prices = aligned_prices.iloc[:, 0] # type: ignore
        else:
            aligned_prices = pd.Series(dtype=float, index=eps_table["end"])

    # Now it's safe to use .isna().any() which returns a single boolean for a Series
    if aligned_prices.isna().any():
        logger.warning(f"Missing price data for some EPS dates for {ticker}; PE ratios may contain NaN")

    # Assign aligned prices and compute PE ratio element-wise
    eps_table["price"] = aligned_prices.values
    eps_table["PE_ratio"] = eps_table["price"] / eps_table["eps"]

    return eps_table


# def get_PB_ratio_data(ticker:str, ticker_price_data: pd.DataFrame, company_facts: dict):
#     equity_table = extract_quarterly_data(company_facts, 'StockholdersEquity', 'USD')
#     shares_df = extract_quarterly_data(company_facts, 'CommonStockSharesOutstanding', 'shares')
