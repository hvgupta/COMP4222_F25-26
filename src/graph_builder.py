from src.logger import logger
from src.feature_lists import (
    ALL_FEATURES,
    HISTORICAL_DATA_FEATURES,
    PB_FEATURES,
    PE_FEATURES,
    ROA_FEATURES,
    ALL_SECTORS_FEATURES,
    CUR_PRICE_FEATURES,
    ALL_GICS_SECTORS,
    CURRENT_FEATURES,
)
from src.market_data_fetcher import (
    SKIPException,
    SP500_COMPANIES,
    TICKER_TO_CIK_MAP,
    fetch_ticker_historical_prices,
    fetch_sec_concepts,
)
from src.company_feature_functions import (
    get_historical_price_features,
    get_PB_ratio_data,
    get_PE_ratio_data,
    get_roa_data,
    get_current_ratio_data,
    get_one_hot_sector,
)

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pandas import Timestamp
from itertools import combinations
from typing import Optional, Tuple, List

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
        self.start_year = start_year - 1
        self.end_year = end_year
        self.company_df = company_df
        self.historical_prices = pd.DataFrame(
            columns=["Date", "Symbol", "Open", "High", "Low", "Close"]
        )
        # this is a subset of the features in order to make the process of creating the graphs easier

        self.features = pd.DataFrame(columns=["Date", "Symbol"] + ALL_FEATURES)

    def _conv_start_end_to_date(self, df: pd.DataFrame):
        df_columns = df.columns.tolist()
        df_columns.remove("start")
        df_columns.remove("end")
        # Collect rows and create DataFrame once (avoids repeated concat + FutureWarning)
        rows = []
        for _, row in df.iterrows():
            start_date = pd.to_datetime(row["start"])
            end_date = pd.to_datetime(row["end"])
            for single_date in pd.date_range(start=start_date, end=end_date, freq="D"):
                new_row = {col: row[col] for col in df_columns}
                new_row["Date"] = single_date
                rows.append(new_row)

        if not rows:
            return pd.DataFrame(columns=["Date"] + df_columns)

        expanded_df = pd.DataFrame(rows)
        # ensure consistent column order and infer better dtypes
        expanded_df = expanded_df[["Date"] + df_columns].infer_objects(copy=False)
        expanded_df["Date"] = pd.to_datetime(expanded_df["Date"])
        return expanded_df

    def get_valid_date_range(
        self,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        df = self.features.copy()
        feature_cols = [col for col in df.columns if col != "Date"]
        valid_rows = df[df[feature_cols].notna().all(axis=1)]

        if valid_rows.empty:
            return None, None  # or raise an error

        earliest = valid_rows["Date"].min()
        latest = valid_rows["Date"].max()
        return earliest, latest

    def _merge_into_company_features(
        self,
        company_features: pd.DataFrame,
        actual_info: pd.DataFrame,
        columns: list[str],
    ):
        expanded_df = self._conv_start_end_to_date(actual_info)
        merged_df = pd.merge(
            company_features,
            expanded_df[["Date"] + columns],
            how="left",
            on="Date",
            suffixes=("_old", "_new"),
        )

        merged_df = merged_df.drop(
            labels=[col for col in merged_df.columns if col.endswith("_old")], axis=1
        )

        merged_df = merged_df.rename(
            {col: col[:-4] for col in merged_df.columns if col.endswith("_new")}, axis=1
        )
        
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])

        return merged_df

    async def _extract_company_info(
        self, row: pd.Series, start_date_str: str, end_date_str: str
    ):
        ticker = row["Symbol"].replace(".", "-")
        prices = await fetch_ticker_historical_prices(
            ticker, start_date_str, end_date_str
        )
        try:
            company_concept = await fetch_sec_concepts(TICKER_TO_CIK_MAP[ticker])
        except Exception as e:
            logger.error(
                f"Got the error while extract sec concepts for ticker: {ticker}, got error: {e}"
            )
            return None, None

        company_features = pd.DataFrame(columns=["Date", "Symbol"] + ALL_FEATURES)

        company_price_features = get_historical_price_features(ticker, prices)
        # company_price_features = company_price_features[]
        company_features[["Date"] + HISTORICAL_DATA_FEATURES] = company_price_features

        try:
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
        except SKIPException as e:
            logger.error(f"The skip error has been raised for ticker {ticker}: {e}")
            return None, None
        except Exception as e:
            logger.error(
                logger.error(f"An error has been raised for ticker {ticker}: {e}")
            )
            raise

        company_features["Symbol"] = ticker
        sector_one_hot = get_one_hot_sector(row["GICS Sector"], ALL_GICS_SECTORS)

        company_features[ALL_SECTORS_FEATURES] = sector_one_hot.values
        company_features[CUR_PRICE_FEATURES] = prices[CUR_PRICE_FEATURES]

        company_features["Date"] = pd.to_datetime(company_features["Date"])

        prices["Symbol"] = ticker

        logger.info(
            f"the current ticker is {ticker}, company features is {'empty' if company_features.empty else 'not empty'}"
        )
        logger.info(
            f"the current ticker is {ticker}, company features is {'None' if company_features.isna().values.all() else 'not None'}"
        )

        return company_features, prices

    def _check_seen_symbols(self):
        output_file = Path(__file__).parent / "features.csv"
        if output_file.exists():
            existing_features = pd.read_csv(output_file, index_col=0)  # Add index_col=0
            seen_symbols = existing_features["Symbol"].unique().tolist()
            print(seen_symbols)
            self.company_df = self.company_df[
                ~self.company_df["Symbol"].isin(seen_symbols)
            ]
            self.features = existing_features
            logger.info(f"Resuming from {len(seen_symbols)} seen symbols.")
        else:
            logger.info("No existing features file found. Starting fresh.")

    async def async_gather_features(self):

        self._check_seen_symbols()

        start_date = Timestamp(year=self.start_year, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        for _, row in self.company_df.iterrows():
            try:
                company_features, historical_prices = await self._extract_company_info(
                    row, start_date_str, end_date_str
                )
            except Exception as e:
                continue

            if company_features is None or historical_prices is None:
                continue

            logger.info(f"insert features for {company_features["Symbol"].unique()[0]}")
            if company_features["Date"].isna().values.any(): # type: ignore
                raise ValueError
            self.features = pd.concat(
                [self.features, company_features], ignore_index=True
            )
            logger.info(
                f"all the unique tickers are {self.features["Symbol"].unique().tolist()}"
            )
            self.historical_prices = pd.concat(
                [self.historical_prices, historical_prices]
            )

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
        edges: list[list[str]] = []
        for i, j in combinations(symbols, 2):
            corr_val = avg_corr_matrix.loc[i, j]
            if pd.notna(corr_val) and corr_val >= self.corr_threshold:  # type: ignore
                edges.append([i, j])

        return edges, avg_corr_matrix

    def conv_edge_index_to_tensor(self, edges: list[list[str]], device):
        symbols = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
        ticker_to_id_map = {ticker: idx for idx, ticker in enumerate(symbols)}
        source_id_list = []
        target_id_list = []

        for edge in edges:
            source_ticker = edge[0]
            target_ticker = edge[1]

            source_id_list.append(ticker_to_id_map[source_ticker])
            target_id_list.append(ticker_to_id_map[target_ticker])

        return torch.Tensor([source_id_list, target_id_list]).to(
            dtype=torch.int64, device=device
        )

    def get_node_features(self, ticker_to_id_map: dict[str, int], date: pd.Timestamp):
        feature_list: List[Optional[pd.DataFrame]] = [None] * len(ticker_to_id_map)
        df = self.features
        for ticker, idx in ticker_to_id_map.items():
            feature_list[idx] = df[(df["Symbol"] == ticker) & (df["Date"] == date)][
                ALL_FEATURES
            ]

        return pd.concat(feature_list, axis=0)

    def get_date_symbols_tripet(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, num_batches: int
    ):
        seen = set()
        df_subset = self.features.copy()
        df_subset = df_subset[
            (df_subset["Date"] >= start_date) & (df_subset["Date"] <= end_date)
        ]["Date", "Symbol"]
        df_subset_shuffled = df_subset.sample(frac=1)
        batch = []

        for row in df_subset_shuffled:
            ...

    async def load_features_csv(self):
        path = Path(__file__).parent / "features.csv"
        if os.path.exists(path):
            self.features = pd.read_csv(path, index_col=0)
            self.features["Date"] = pd.to_datetime(
                self.features["Date"], format="mixed"
            )
            self.features.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.features.dropna(inplace=True)
            self.historical_prices = self.features[
                ["Date", "Symbol", "Close", "High", "Low", "Open"]
            ]
        else:
            await self.async_gather_features()
