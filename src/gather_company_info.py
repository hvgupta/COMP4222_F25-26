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
    ticker: str, ticker_price_data: pd.DataFrame, company_facts: dict
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

    eps_table = clean_eps_table(eps_table)
    logger.info(f"Extracted {len(eps_table)} EPS points for {ticker}")

    stock_price = ticker_price_data.loc[eps_table["end"], "Close"].reset_index(drop=True)

    eps_table["PE_ratio"] = stock_price / eps_table["eps"]
    eps_table["price"] = stock_price

    return eps_table


# def get_PB_ratio_data(ticker:str, ticker_price_data: pd.DataFrame, company_facts: dict):
#     equity_table = extract_quarterly_data(company_facts, 'StockholdersEquity', 'USD')
#     shares_df = extract_quarterly_data(company_facts, 'CommonStockSharesOutstanding', 'shares')
