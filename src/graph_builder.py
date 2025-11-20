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
from numpy.typing import NDArray
from itertools import combinations, permutations
from typing import Optional, Tuple, Dict, List, TypedDict

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
        self, row: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp
    ):
        ticker = row["Symbol"].replace(".", "-")
        prices = await fetch_ticker_historical_prices(
            ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
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

        if (company_features["Date"].min() >= start_date + pd.DateOffset(years=1)) or (
            (company_features["Date"].max() <= end_date - pd.DateOffset(years=1))
        ):
            return None, None

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

        for _, row in self.company_df.iterrows():
            try:
                company_features, historical_prices = await self._extract_company_info(
                    row, start_date, end_date
                )
            except Exception as e:
                continue

            if company_features is None or historical_prices is None:
                continue

            logger.info(f"insert features for {company_features["Symbol"].unique()[0]}")
            if company_features["Date"].isna().values.any():  # type: ignore
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

        sym_start_end_df = (
            self.features[["Date", "Symbol"]]
            .groupby("Symbol")
            .describe()["Date"][["min", "max"]]
        )

        latest_start_date: pd.Timestamp = sym_start_end_df["min"].max()
        earliest_end_date: pd.Timestamp = sym_start_end_df["max"].min()

        self.features = self.features[
            (self.features["Date"] >= latest_start_date)
            & (self.features["Date"] <= earliest_end_date)
        ]

        return self.features

    def _get_edges(self, rolling_corr: pd.DataFrame, symbols: list[str], device):
        avg_corr_matrix = rolling_corr.groupby(level=1).mean()

        edges: list[list[str]] = []
        for i, j in combinations(symbols, 2):
            corr_val = avg_corr_matrix.loc[i, j]
            if pd.notna(corr_val) and corr_val >= self.corr_threshold:  # type: ignore
                edges.append([i, j])

        return self._conv_edge_index_to_tensor(edges, device)

    def _conv_edge_index_to_tensor(self, edges: list[list[str]], device):
        source_id_list = []
        target_id_list = []

        for edge in edges:
            source_ticker = edge[0]
            target_ticker = edge[1]

            source_id_list.append(self.ticker_to_id_map[source_ticker])
            target_id_list.append(self.ticker_to_id_map[target_ticker])

        return torch.Tensor([source_id_list, target_id_list]).to(
            dtype=torch.int64, device=device
        )

    def _get_all_node_features_and_next_day_pct1_at_date(
        self, date: pd.Timestamp, device
    ):
        feature_list: List[NDArray] = [] * len(self.ticker_to_id_map)
        df = self.features[self.features["Date"] == date]

        for ticker, idx in self.ticker_to_id_map.items():
            feature_list[idx] = df[df["Symbol"] == ticker][ALL_FEATURES].to_numpy()

        return torch.tensor(feature_list, device=device)

    def _get_next_day_pct_change(self, date: pd.Timestamp, device):
        next_day = date + pd.Timedelta(days=1)
        pct_list: List[float] = [0] * len(self.ticker_to_id_map)

        df = self.features[self.features["Date"] == next_day]

        for ticker, idx in self.ticker_to_id_map.items():
            pct_list[idx] = df[df["Symbol"] == ticker]["PCT-1"].tolist()[0]

        return torch.tensor(pct_list, device=device)

    def _load_date_specific_info(
        self,
        all_dates: List[pd.Timestamp],
        column: str,
        symbols: list[str],
        device,
    ) -> Dict[pd.Timestamp, Dict[str, torch.Tensor]]:

        pivot = self.features.pivot(
            index="Date", columns="Symbol", values=column
        ).sort_index()

        rolling_corr = pivot.rolling(self.window_size, min_periods=1).corr()

        date_info_map = {}

        for date in all_dates:
            rolling_corr_subset = rolling_corr[rolling_corr["Date"] <= date]
            date_info_map[date] = {
                "edge_index": self._get_edges(rolling_corr_subset, symbols, device),
                "node_features": self._get_all_node_features_and_next_day_pct1_at_date(
                    date, device
                ),
                "next_day_pct1": self._get_next_day_pct_change(date, device),
            }

        return date_info_map

    def _train_test_split(self, df: pd.DataFrame, train_frac: float):
        train_data_points = df.groupby("Date").sample(frac=train_frac)
        test_data_points = df.loc[~df.index.isin(train_data_points.index)]

        return train_data_points, test_data_points

    def load_dataset(
        self,
        days_skip: int = 15,
        column: str = "Close",
        train_frac: float = 0.8,
        *,
        device,
    ):

        latest_start_date: pd.Timestamp = self.features["Date"].min()
        earliest_end_date: pd.Timestamp = self.features["Date"].max()

        usable_end_date = earliest_end_date - pd.DateOffset(days=1)
        # we cant take the last date to predict the next day since some nodes may be missing (more difficult to deal with so is being ignored)

        triplet_df = pd.DataFrame(columns=["Date", "src_node", "trgt_node"])
        symbols: list[str] = self.features["Symbol"].unique().tolist()

        self.ticker_to_id_map = {ticker: idx for idx, ticker in enumerate(symbols)}

        all_perms = [
            {"src_node": src, "trgt_node": trgt}
            for src, trgt in permutations(symbols, 2)
        ]

        all_dates = pd.date_range(
            latest_start_date, usable_end_date, freq=pd.Timedelta(days=days_skip)
        )

        # Create a DataFrame from all_perms
        perms_df = pd.DataFrame(all_perms)

        # Create a DataFrame from all_dates
        dates_df = pd.DataFrame({"Date": all_dates})

        # Create Cartesian product using merge with a dummy key
        perms_df["key"] = 1
        dates_df["key"] = 1

        # Merge to get all combinations: each date with every permutation
        triplet_df = pd.merge(dates_df, perms_df, on="key").drop("key", axis=1)

        return *self._train_test_split(
            triplet_df, train_frac
        ), self._load_date_specific_info(
            all_dates, column, symbols, device  # type: ignore
        )

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
