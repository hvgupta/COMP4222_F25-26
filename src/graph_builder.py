from .company_feature_functions import *
from .feature_lists import *

import pandas as pd
from pathlib import Path
from random import sample
from pandas import Timestamp
from itertools import combinations

# ======= HYPER-PARAMETERS ======

WINDOW_SIZE = 5
"Number of Trading Days considered for correlation calculation"  # basically take the average over the window size

CORRELATION_THRESHOLD = 0.8
"Minimum correlation value to consider an edge between two tickers"

START_YEAR = 2020

END_YEAR = 2025

# ===============================


class GraphManager:
    def __init__(
        self,
        window_size=WINDOW_SIZE,
        corr_threshold=CORRELATION_THRESHOLD,
        start_year=START_YEAR,
        end_year=END_YEAR,
        company_df=SP500_COMPANIES,
    ):
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
        df_columns = df.columns.tolist()
        df_columns.remove("start")
        df_columns.remove("end")
        expanded_df = pd.DataFrame(columns=["date"] + df_columns)

        for i, row in df.iterrows():
            start_date = pd.to_datetime(row["start"])
            end_date = pd.to_datetime(row["end"])
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")

            for single_date in date_range:
                new_row = {col: row[col] for col in df_columns}
                new_row["date"] = single_date
                expanded_df = pd.concat(
                    [expanded_df, pd.DataFrame([new_row])], ignore_index=True
                )
        return expanded_df

    def _merge_into_company_features(
        self,
        company_features: pd.DataFrame,
        actual_info: pd.DataFrame,
        columns: list[str],
    ):
        expanded_df = self._conv_start_end_to_date(actual_info)
        company_features[columns] = expanded_df[
            expanded_df["date"].isin(company_features["Date"])
        ][columns].reset_index(drop=True)
        return company_features

    async def _extract_company_info(
        self, row: pd.Series, start_date_str: str, end_date_str: str
    ):
        ticker = row["Symbol"]
        prices = await fetch_ticker_historical_prices(
            ticker, start_date_str, end_date_str
        )

        self.historical_prices.loc[:, pd.IndexSlice[:, ticker]] = prices  # type: ignore

        company_concept = await fetch_sec_concepts(TICKER_TO_CIK_MAP[ticker])

        company_features = pd.DataFrame(columns=ALL_FEATURES)

        company_price_features = get_historical_price_features(ticker, prices)
        # company_price_features = company_price_features[]
        company_features[["Date"] + HISTORICAL_DATA_FEATURES] = company_price_features

        company_PE_ratio_df = await get_PE_ratio_data(
            ticker, prices, company_concept["facts"], self.start_year, self.end_year
        )
        company_features = self._merge_into_company_features(
            company_features, company_PE_ratio_df, PE_FEATURES
        )

        company_PB_ratio_df = await get_PB_ratio_data(
            ticker, prices, company_concept["facts"], self.start_year, self.end_year
        )
        company_features = self._merge_into_company_features(
            company_features, company_PB_ratio_df, PB_FEATURES
        )

        company_ROA_df = await get_roa_data(
            ticker, company_concept["facts"], self.start_year, self.end_year
        )
        company_features = self._merge_into_company_features(
            company_features, company_ROA_df, ROA_FEATURES
        )

        company_current_ratio_df = await get_current_ratio_data(
            ticker, company_concept["facts"], self.start_year, self.end_year
        )
        company_features = self._merge_into_company_features(
            company_features, company_current_ratio_df, CURRENT_FEATURES
        )

        company_features["Symbol"] = ticker
        sector_one_hot = get_one_hot_sector(row["GICS Sector"], ALL_GICS_SECTORS)
        company_features[sector_one_hot.index] = sector_one_hot.values
        company_features[CUR_PRICE_FEATURES] = prices[CUR_PRICE_FEATURES]

        return company_features

    def _check_seen_symbols(self):
        output_file = Path(__file__).parent / "features_test_output.csv"
        if output_file.exists():
            existing_features = pd.read_csv(output_file)
            seen_symbols = existing_features["Symbol"].unique().tolist()
            self.company_df = self.company_df[
                ~self.company_df["Symbol"].isin(seen_symbols)
            ]
            self.features = existing_features
            print(f"Resuming from {len(seen_symbols)} seen symbols.")
        else:
            print("No existing features file found. Starting fresh.")

    async def async_gather_features(self, batch_size: int = 10):

        self._check_seen_symbols()

        start_date = Timestamp(year=self.start_year - 1, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        all_feature_tasks = [
            self._extract_company_info(row, start_date_str, end_date_str)
            for _, row in self.company_df.iterrows()
        ]

        for i in range(0, len(all_feature_tasks), batch_size):
            batch_tasks = all_feature_tasks[i : i + batch_size]
            batch_company_features = await asyncio.gather(
                *batch_tasks, return_exceptions=True
            )

            for company_features in batch_company_features:
                if not isinstance(company_features, pd.DataFrame):
                    logger.error(f"Error fetching company features: {company_features}")
                    continue
                self.features = pd.concat(
                    [self.features, company_features], ignore_index=True
                )

            await asyncio.sleep(1)  # brief pause between batches to avoid rate limits

        return self.features

    def build_graph(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, column: str
    ):
        """
        Build a correlation graph using rolling-window average correlations.

        Args:
            start_date (Timestamp): The start of the date range.
            end_date (Timestamp): The end of the date range.
            column (str): Price column ("Close", "Open", "Return", etc.)

        Returns:
            edges: list of (stock_i, stock_j, avg_corr)
            corr_matrix: DataFrame of average correlations
        """

        # ─────────────────────────────────────────────
        # STEP 1: Filter by date + pivot to wide format
        # ─────────────────────────────────────────────
        df = self.historical_prices
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        pivot = df.pivot(index="Date", columns="Symbol", values=column).sort_index()

        # Optionally: convert prices → daily returns
        # pivot = pivot.pct_change().dropna()

        symbols = pivot.columns.tolist()

        # ─────────────────────────────────────────────
        # STEP 2: Compute rolling correlations for pairs
        # ─────────────────────────────────────────────
        # This creates a multi-index correlation DataFrame
        rolling_corr = pivot.rolling(self.window_size).corr()

        # ─────────────────────────────────────────────
        # STEP 3: Average correlation across all windows
        # ─────────────────────────────────────────────
        avg_corr_matrix = rolling_corr.groupby(level=1).mean()

        # Now avg_corr_matrix is a square matrix like:
        #         AAPL   MSFT   NVDA
        # AAPL    1.0    0.65   0.52
        # MSFT    0.65   1.0    0.58
        # NVDA    0.52   0.58   1.0

        # ─────────────────────────────────────────────
        # STEP 4: Build edge list using threshold
        # ─────────────────────────────────────────────
        edges = []
        for i, j in combinations(symbols, 2):
            corr_val = avg_corr_matrix.loc[i, j]
            if pd.notna(corr_val) and corr_val >= self.corr_threshold:  # type: ignore
                edges.append((i, j, float(corr_val)))  # type: ignore

        return edges, avg_corr_matrix
    
    def get_dataset(self, end_year: int):
        ...
