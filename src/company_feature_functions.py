from .logger import logger
from .market_data_fetcher import *

import talib
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

    eps_table = clean_period_table(eps_table, start_year, end_year, "eps")
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


def get_roa_data(ticker: str, company_facts: dict, start_year: int, end_year: int):
    logger.info(f"Computing ROA for {ticker} from {start_year} to {end_year}")
    net_df = extract_quarterly_data(company_facts, "NetIncomeLoss", "USD")
    assets_df = extract_quarterly_data(company_facts, "Assets", "USD")

    net_df["start"] = pd.to_datetime(net_df["start"], errors="coerce")  # type: ignore
    net_df["end"] = pd.to_datetime(net_df["end"], errors="coerce")  # type: ignore
    assets_df["end"] = pd.to_datetime(assets_df["end"], errors="coerce")  # type: ignore

    cleaned_net_df = clean_period_table(net_df, start_year, end_year, "net_q")
    cleaned_assets_df = clean_instance_tables(assets_df, start_year, end_year, "assets")

    combined_df = pd.merge_asof(
        cleaned_net_df.sort_values("end"),
        cleaned_assets_df.sort_values("end"),
        on="end",
        by=["start", "fp"],
        direction="backward",
        suffixes=("_net", "_assets"),
    )

    combined_df["roa"] = combined_df["net_q"] / combined_df["assets"]
    logger.info(f"Computed ROA table with {len(combined_df)} rows for {ticker}")
    return combined_df


def get_current_ratio_data(
    ticker: str,
    company_facts: dict,
    start_year: int,
    end_year: int,
):
    logger.info(
        f"Computing ROA and Current Ratio for {ticker} from {start_year} to {end_year}"
    )

    assets_current_df = extract_quarterly_data(company_facts, "AssetsCurrent", "USD")
    liab_current_df = extract_quarterly_data(company_facts, "LiabilitiesCurrent", "USD")

    assets_current_df["end"] = pd.to_datetime(assets_current_df["end"], errors="coerce")  # type: ignore
    liab_current_df["end"] = pd.to_datetime(liab_current_df["end"], errors="coerce")  # type: ignore

    cleaned_cur_assets = clean_instance_tables(
        assets_current_df,
        start_year=start_year,
        end_year=end_year,
        quantity_name="assets_current",
    )
    cleaned_liab_current = clean_instance_tables(
        liab_current_df,
        start_year=start_year,
        end_year=end_year,
        quantity_name="liabilities_current",
    )

    merged_df = pd.merge_asof(
        cleaned_cur_assets.sort_values("end"),
        cleaned_liab_current.sort_values("end"),
        on="end",
        by=["start", "fp"],
        direction="backward",
        suffixes=("_assets", "_liab"),
    )
    merged_df["current_ratio"] = (
        merged_df["assets_current"] / merged_df["liabilities_current"]
    )
    logger.info(f"Computed Current Ratio table with {len(merged_df)} rows for {ticker}")
    return merged_df


def get_historical_price_features(ticker: str, ticker_price_data: pd.DataFrame):
    logger.info(f"Computing historical price features for {ticker}")

    price_df = ticker_price_data.copy()
    price_features_df = pd.DataFrame(
        columns=[
            "date",
            "PCT-1",
            "PCT-5",
            "PCT-10",
            "PCT-15",
            "PCT-20",
            "MOM-5",
            "MOM-10",
            "MOM-15",
            "MOM-20",
            "NATR-5",
            "NATR-10",
            "NATR-15",
            "NATR-20",
        ]
    )

    close_numpy = price_df["Close"].to_numpy().reshape(-1)
    high_numpy = price_df["High"].to_numpy().reshape(-1)
    low_numpy = price_df["Low"].to_numpy().reshape(-1)

    pct1 = talib.ROCP(close_numpy, timeperiod=1)
    pct5 = talib.ROCP(close_numpy, timeperiod=5)
    pct10 = talib.ROCP(close_numpy, timeperiod=10)
    pct15 = talib.ROCP(close_numpy, timeperiod=15)
    pct20 = talib.ROCP(close_numpy, timeperiod=20)

    mom5 = talib.MOM(close_numpy, timeperiod=5)
    mom10 = talib.MOM(close_numpy, timeperiod=10)
    mom15 = talib.MOM(close_numpy, timeperiod=15)
    mom20 = talib.MOM(close_numpy, timeperiod=20)

    natr5 = talib.NATR(high_numpy, low_numpy, close_numpy, timeperiod=5)
    natr10 = talib.NATR(high_numpy, low_numpy, close_numpy, timeperiod=10)
    natr15 = talib.NATR(high_numpy, low_numpy, close_numpy, timeperiod=15)
    natr20 = talib.NATR(high_numpy, low_numpy, close_numpy, timeperiod=20)

    price_features_df["date"] = price_df.index

    price_features_df["PCT-1"] = pct1
    price_features_df["PCT-5"] = pct5
    price_features_df["PCT-10"] = pct10
    price_features_df["PCT-15"] = pct15
    price_features_df["PCT-20"] = pct20

    price_features_df["MOM-5"] = mom5
    price_features_df["MOM-10"] = mom10
    price_features_df["MOM-15"] = mom15
    price_features_df["MOM-20"] = mom20

    price_features_df["NATR-5"] = natr5
    price_features_df["NATR-10"] = natr10
    price_features_df["NATR-15"] = natr15
    price_features_df["NATR-20"] = natr20

    logger.info(f"Computed historical price features for {ticker}")

    return price_features_df

def get_one_hot_sector(ticker_sector: str, all_sectors: list[str]) -> pd.Series:
    logger.info(f"Generating one-hot encoding for sector: {ticker_sector}")
    one_hot_series = pd.Series(0, index=all_sectors)
    if ticker_sector in all_sectors:
        one_hot_series[ticker_sector] = 1
    else:
        logger.warning(f"Sector {ticker_sector} not found in all sectors list")
    return one_hot_series


def get_profit_margin_data(
    ticker: str,
    company_facts: dict,
    start_year: int,
    end_year: int,
):
    """
    Compute profit margin per quarter using SEC facts.
    Profit Margin = Net Income / Revenue
    Returns DataFrame with columns: start, end, year, quarter, net_income, revenue, profit_margin
    """
    logger.info(f"Computing Profit Margin for {ticker} from {start_year} to {end_year}")
    
    net_df = extract_quarterly_data(company_facts, "NetIncomeLoss", "USD")
    revenue_df = extract_quarterly_data(company_facts, "Revenues", "USD")
    
    if net_df.empty:
        logger.warning(f"No NetIncomeLoss data for {ticker}")
        return pd.DataFrame()
    if revenue_df.empty:
        logger.warning(f"No Revenues data for {ticker}")
        return pd.DataFrame()
    
    # normalize datetimes
    net_df["start"] = pd.to_datetime(net_df["start"], errors="coerce")  # type: ignore
    net_df["end"] = pd.to_datetime(net_df["end"], errors="coerce")  # type: ignore
    revenue_df["start"] = pd.to_datetime(revenue_df["start"], errors="coerce")  # type: ignore
    revenue_df["end"] = pd.to_datetime(revenue_df["end"], errors="coerce")  # type: ignore
    
    # clean period tables (quarterly flow metrics)
    cleaned_net = clean_period_table(net_df, start_year, end_year, "net_income")
    cleaned_revenue = clean_period_table(revenue_df, start_year, end_year, "revenue")
    
    if cleaned_net.empty or cleaned_revenue.empty:
        logger.warning(f"Cleaned tables are empty for {ticker}")
        return pd.DataFrame()
    
    # merge on end date and fp
    merged_df = pd.merge(
        cleaned_net,
        cleaned_revenue,
        on=["end", "start", "fp"],
        how="inner",
        suffixes=("_net", "_rev")
    )
    
    # compute profit margin
    merged_df["profit_margin"] = merged_df["net_income"] / merged_df["revenue"].replace({0: pd.NA})
    
    merged_df["year"] = merged_df["end"].dt.year
    merged_df["quarter"] = merged_df["fp"]
    
    logger.info(f"Computed Profit Margin table with {len(merged_df)} rows for {ticker}")
    return merged_df


def get_interest_coverage_ratio_data(
    ticker: str,
    company_facts: dict,
    start_year: int,
    end_year: int,
):
    """
    Compute Interest Coverage Ratio per quarter using SEC facts.
    Interest Coverage Ratio = EBIT / Interest Expense
    Where EBIT = Operating Income (or Net Income + Interest Expense + Tax if not available)
    
    Returns DataFrame with columns: start, end, year, quarter, operating_income, interest_expense, interest_coverage_ratio
    """
    logger.info(f"Computing Interest Coverage Ratio for {ticker} from {start_year} to {end_year}")
    
    # Try to get OperatingIncomeLoss first (EBIT proxy)
    operating_income_df = extract_quarterly_data(company_facts, "OperatingIncomeLoss", "USD")
    interest_expense_df = extract_quarterly_data(company_facts, "InterestExpense", "USD")
    
    if operating_income_df.empty:
        logger.warning(f"No OperatingIncomeLoss data for {ticker}, trying alternative calculation")
        # Alternative: try to compute EBIT from net income + interest + tax
        net_income_df = extract_quarterly_data(company_facts, "NetIncomeLoss", "USD")
        tax_df = extract_quarterly_data(company_facts, "IncomeTaxExpenseBenefit", "USD")
        
        if net_income_df.empty or interest_expense_df.empty:
            logger.warning(f"Insufficient data to compute Interest Coverage Ratio for {ticker}")
            return pd.DataFrame()
        
        # Normalize and clean
        net_income_df["start"] = pd.to_datetime(net_income_df["start"], errors="coerce")  # type: ignore
        net_income_df["end"] = pd.to_datetime(net_income_df["end"], errors="coerce")  # type: ignore
        interest_expense_df["start"] = pd.to_datetime(interest_expense_df["start"], errors="coerce")  # type: ignore
        interest_expense_df["end"] = pd.to_datetime(interest_expense_df["end"], errors="coerce")  # type: ignore
        
        cleaned_net = clean_period_table(net_income_df, start_year, end_year, "net_income")
        cleaned_interest = clean_period_table(interest_expense_df, start_year, end_year, "interest_expense")
        
        if not tax_df.empty:
            tax_df["start"] = pd.to_datetime(tax_df["start"], errors="coerce")  # type: ignore
            tax_df["end"] = pd.to_datetime(tax_df["end"], errors="coerce")  # type: ignore
            cleaned_tax = clean_period_table(tax_df, start_year, end_year, "tax_expense")
            
            # Merge all three
            merged_df = pd.merge(cleaned_net, cleaned_interest, on=["end", "start", "fp"], how="inner")
            merged_df = pd.merge(merged_df, cleaned_tax, on=["end", "start", "fp"], how="left")
            
            # EBIT = Net Income + Interest + Tax
            merged_df["operating_income"] = (
                merged_df["net_income"] + 
                merged_df["interest_expense"] + 
                merged_df["tax_expense"].fillna(0)
            )
        else:
            # Without tax data: EBIT ≈ Net Income + Interest
            merged_df = pd.merge(cleaned_net, cleaned_interest, on=["end", "start", "fp"], how="inner")
            merged_df["operating_income"] = merged_df["net_income"] + merged_df["interest_expense"]
    
    else:
        # Use operating income directly
        if interest_expense_df.empty:
            logger.warning(f"No InterestExpense data for {ticker}")
            return pd.DataFrame()
        
        operating_income_df["start"] = pd.to_datetime(operating_income_df["start"], errors="coerce")  # type: ignore
        operating_income_df["end"] = pd.to_datetime(operating_income_df["end"], errors="coerce")  # type: ignore
        interest_expense_df["start"] = pd.to_datetime(interest_expense_df["start"], errors="coerce")  # type: ignore
        interest_expense_df["end"] = pd.to_datetime(interest_expense_df["end"], errors="coerce")  # type: ignore
        
        cleaned_operating = clean_period_table(operating_income_df, start_year, end_year, "operating_income")
        cleaned_interest = clean_period_table(interest_expense_df, start_year, end_year, "interest_expense")
        
        if cleaned_operating.empty or cleaned_interest.empty:
            logger.warning(f"Cleaned tables are empty for {ticker}")
            return pd.DataFrame()
        
        merged_df = pd.merge(
            cleaned_operating,
            cleaned_interest,
            on=["end", "start", "fp"],
            how="inner"
        )
    
    # Compute Interest Coverage Ratio
    merged_df["interest_coverage_ratio"] = (
        merged_df["operating_income"] / merged_df["interest_expense"].replace({0: pd.NA})
    )
    
    merged_df["year"] = merged_df["end"].dt.year
    merged_df["quarter"] = merged_df["fp"]
    
    logger.info(f"Computed Interest Coverage Ratio table with {len(merged_df)} rows for {ticker}")
    return merged_df