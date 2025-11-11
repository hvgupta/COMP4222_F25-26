from .logger import logger
from .market_data_fetcher import *

import pandas as pd


def _price_align_and_compute_ratio(
    price_data: pd.DataFrame,
    financial_data: pd.DataFrame,
    value_col: str,
    ratio_col: str,
):
    price_df = price_data.copy()
    price_df.index = pd.to_datetime(price_df.index, errors="coerce")
    price_series = price_df["Close"].sort_index()

    try:
        aligned_prices = price_series.reindex(financial_data["end"], method="ffill")
    except Exception:
        aligned_prices = price_series.reindex(financial_data["end"])

    logger.info(f"Aligned prices to financial data for ratio calculation: {ratio_col}")

    if isinstance(aligned_prices, pd.DataFrame):
        if not aligned_prices.empty:
            aligned_prices = aligned_prices.iloc[:, 0]  # type: ignore
        else:
            aligned_prices = pd.Series(dtype=float, index=financial_data["end"])

    if aligned_prices.isna().any():
        logger.warning(
            f"Missing price data for some financial dates; {ratio_col} may contain NaN"
        )

    financial_data["price"] = aligned_prices.values
    financial_data[ratio_col] = financial_data["price"] / financial_data[value_col]
    return financial_data


def get_PE_ratio_data(
    ticker: str,
    ticker_price_data: pd.DataFrame,
    company_facts: dict,
    start_year: int,
    end_year: int,
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

    return _price_align_and_compute_ratio(
        ticker_price_data, eps_table, value_col="eps", ratio_col="PE_ratio"
    )


def get_PB_ratio_data(
    ticker: str,
    ticker_price_data: pd.DataFrame,
    company_facts: dict,
    start_year: int,
    end_year: int,
):
    logger.info(f"Extracting PB ratio data for {ticker}")
    equity_table = extract_quarterly_data(company_facts, "StockholdersEquity", "USD")
    logger.info(f"Extracted {len(equity_table)} equity points for {ticker}")

    shares_table = extract_quarterly_data(
        company_facts, "CommonStockSharesOutstanding", "shares"
    )
    logger.info(f"Extracted {len(shares_table)} shares outstanding points for {ticker}")

    standardised_equity = clean_instance_tables(
        equity_table,
        start_year=start_year,
        end_year=end_year,
        quantity_name="stockHolder_equity",
    )
    logger.info(
        f"Standardised equity to {len(standardised_equity)} points for {ticker}"
    )

    standardised_shares = clean_instance_tables(
        shares_table,
        start_year=start_year,
        end_year=end_year,
        quantity_name="shareQuantity",
    )

    logger.info(
        f"Standardised shares to {len(standardised_shares)} points for {ticker}"
    )

    equity_per_share_table = pd.merge_asof(
        standardised_equity.sort_values("end"),
        standardised_shares.sort_values("end"),
        on="end",
        direction="backward",
        suffixes=("_equity", "_shares"),
    )
    equity_per_share_table["equity_per_share"] = (
        equity_per_share_table["stockHolder_equity"]
        / equity_per_share_table["shareQuantity"]
    )

    logger.info(f"Computed equity per share for {ticker}")

    return _price_align_and_compute_ratio(
        ticker_price_data,
        equity_per_share_table,
        value_col="equity_per_share",
        ratio_col="PB_ratio",
    )
