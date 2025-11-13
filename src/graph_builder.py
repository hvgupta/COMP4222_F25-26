from .market_data_fetcher import *
from .company_feature_functions import *

import pandas as pd
from pandas import Timestamp, MultiIndex

# ======= HYPER-PARAMETERS ======

WINDOW_SIZE = 5
"Number of Trading Days considered for correlation calculation"  # basically take the average over the window size

CORRELATION_THRESHOLD = 0.8
"Minimum correlation value to consider an edge between two tickers"

START_YEAR = 2020

END_YEAR = 2025

# ===============================

TICKER_TO_CIK_MAP = fetch_ticker_to_cik_map()
SP500_COMPANIES = fetch_sp500_companies()

HISTORICAL_DATA_FEATURES = [
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

ALL_FEATURES = [
    "Date",
    "Symbol",
    "Open",
    "Low",
    "High",
    "Close",
    "PCT-1",
    *HISTORICAL_DATA_FEATURES,
    "PE-Ratio",
    "PB-Ratio",
    "ROA",
    "Current-Ratio",
    *[
        "Sector_" + sector
        for sector in SP500_COMPANIES["GICS Sector"].unique().tolist()
    ],
]


class GraphManager:
    def __init__(
        self,
        eod_data_csv: str,
        window_size=WINDOW_SIZE,
        corr_threshold=CORRELATION_THRESHOLD,
        start_year=START_YEAR,
        end_year=END_YEAR,
        company_df=SP500_COMPANIES,
    ):
        self.historical_eod_data = pd.read_csv(eod_data_csv)
        self.window_size = window_size
        self.corr_threshold = corr_threshold
        self.start_year = start_year
        self.end_year = end_year
        self.company_df = company_df
        self.historical_prices = pd.DataFrame(
            columns=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
        )
        # this is a subset of the features in order to make the process of creating the graphs easier

        self.features = pd.DataFrame(columns=ALL_FEATURES)
        
    def _conv_start_end_to_date(self, df: pd.DataFrame):
        ...

    def gather_features(self):
        start_date = Timestamp(year=self.start_year - 1, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        for ticker in self.company_df["Symbol"]:
            prices = fetch_ticker_historical_prices(
                ticker, start_date_str, end_date_str
            )

            self.historical_prices.loc[:, pd.IndexSlice[:, ticker]] = prices  # type: ignore

            company_concept = fetch_sec_concepts(TICKER_TO_CIK_MAP[ticker])

            company_features = pd.DataFrame(columns=ALL_FEATURES)

            company_price_features = get_historical_price_features(ticker, prices)
            # company_price_features = company_price_features[]
            company_features[["Date", *HISTORICAL_DATA_FEATURES]] = company_price_features

            company_PE_ratio_df = get_PE_ratio_data(
                ticker, prices, company_concept["facts"], self.start_year, self.end_year
            )
            
            
            company_PB_ratio_df = get_PB_ratio_data(
                ticker, prices, company_concept["facts"], self.start_year, self.end_year
            )
            company_ROA = get_roa_data(
                ticker, company_concept["facts"], self.start_year, self.end_year
            )
            company_current_ratio = get_current_ratio_data(
                ticker, company_concept["facts"], self.start_year, self.end_year
            )
