from .market_data_fetcher import *

import pandas as pd
from pandas import Timestamp

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


class GraphManager:
    def __init__(
        self,
        eod_data_csv: str,
        window_size=WINDOW_SIZE,
        corr_threshold=CORRELATION_THRESHOLD,
        start_year=START_YEAR,
        end_year=END_YEAR,
        company_list=SP500_COMPANIES,
    ):
        self.historical_eod_data = pd.read_csv(eod_data_csv)
        self.window_size = window_size
        self.corr_threshold = corr_threshold
        self.start_year = start_year
        self.end_year = end_year
        self.company_list = company_list

    def gather_features(self):
        start_date = Timestamp(year=self.start_year - 1, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        for ticker in self.company_list["Symbol"]:
            prices = fetch_ticker_historical_prices(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            company_concept = fetch_sec_concepts(TICKER_TO_CIK_MAP[ticker])