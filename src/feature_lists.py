from src.market_data_fetcher import *

TICKER_TO_CIK_MAP = fetch_ticker_to_cik_map()
SP500_COMPANIES = fetch_sp500_companies()

ALL_GICS_SECTORS = SP500_COMPANIES["GICS Sector"].unique().tolist()

HISTORICAL_DATA_FEATURES = [
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

PE_FEATURES = [
    "trailing_eps",
    "trailing_PE_ratio",
    "trailing_one_year_eps",
    "trailing_one_year_PE_ratio",
]

PB_FEATURES = [
    "trailing_eqps",
    "trailing_PB_ratio",
    "trailing_one_year_eqps",
    "trailing_one_year_PB_ratio",
]

ROA_FEATURES = ["trailing_roa", "one_year_avg_trailing_roa"]

CURRENT_FEATURES = [
    "trailing_CR",
    "one_year_avg_trailing_CR",
]

CUR_PRICE_FEATURES = [
    "Open",
    "Low",
    "High",
    "Close",
    "Volume"
]

ALL_SECTORS_FEATURES = ["Sector_" + sector for sector in ALL_GICS_SECTORS]

ALL_FEATURES: list[str] = [
    "Date",
    "Symbol",
    *CUR_PRICE_FEATURES,
    *HISTORICAL_DATA_FEATURES,
    *PE_FEATURES,
    *PB_FEATURES,
    *ROA_FEATURES,
    *CURRENT_FEATURES,
    *ALL_SECTORS_FEATURES,
]
