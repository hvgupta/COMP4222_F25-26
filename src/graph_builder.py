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
import joblib
import pickle
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from random import sample
from pandas import Timestamp
from typing import Optional, Tuple
from itertools import permutations
from sklearn.preprocessing import StandardScaler

# STANDARDIZE using sklearn for easy save/load

# ======= HYPER-PARAMETERS ======

WINDOW_SIZE = 5
"Number of Trading Days considered for correlation calculation" 
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
        training_fraction=0.8,
    ):
        self.window_size = window_size
        self.corr_threshold = corr_threshold
        self.start_year = start_year - 1
        self.end_year = end_year
        self.company_df = company_df
        # this is a subset of the features in order to make the process of creating the graphs easier

        self.features = pd.DataFrame(columns=["Date", "Symbol"] + ALL_FEATURES)
        self.test_features_per_date = {}
        self.train_frac = training_fraction

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
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]: ...

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
        for feature in ALL_SECTORS_FEATURES:
            company_features[feature] = company_features[feature].astype(np.float32)

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
            self.company_df = self.company_df[
                ~self.company_df["Symbol"].isin(seen_symbols)
            ]
            self.features = existing_features
            logger.info(f"Resuming from {len(seen_symbols)} seen symbols.")
        else:
            logger.info("No existing features file found. Starting fresh.")

    async def async_gather_features(self, batch_size=10):
        self._check_seen_symbols()

        start_date = Timestamp(year=self.start_year, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        # Process companies concurrently in batches
        feature_batches = []
        price_batches = []

        for i in range(0, len(self.company_df), batch_size):
            batch = self.company_df.iloc[i : i + batch_size]

            # Create tasks for concurrent execution
            tasks = [
                self._extract_company_info(row, start_date, end_date)
                for _, row in batch.iterrows()
            ]

            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.error(
                        f"Error processing {batch.iloc[idx]['Symbol']}: {result}"
                    )
                    continue

                company_features, historical_prices = result

                if company_features is None or historical_prices is None:
                    continue

                if company_features["Date"].isna().values.any():  # type: ignore
                    continue

                feature_batches.append(company_features)
                price_batches.append(historical_prices)

        # Concatenate all at once instead of iteratively
        if feature_batches:
            self.features = pd.concat(
                [self.features] + feature_batches, ignore_index=True
            )

        # Vectorized date filtering
        self.features["Date"] = pd.to_datetime(self.features["Date"])
        self.features.dropna(inplace=True)

        # Calculate date range for each symbol
        date_stats = self.features.groupby("Symbol")["Date"].agg(["min", "max"])
        date_stats["range"] = (date_stats["max"] - date_stats["min"]).dt.days

        # Get the maximum range
        max_range = date_stats["range"].max()

        # Filter companies with maximum range
        companies_with_max_range = date_stats[
            date_stats["range"] == max_range
        ].index.tolist()

        # Filter the dataframe
        self.features = self.features[
            self.features["Symbol"].isin(companies_with_max_range)
        ]

        self.features.sort_values(by=["Date", "Symbol"], inplace=True)

        # In async_gather_features, after filtering by max range:
        # Also ensure no missing dates
        pivot_test = self.features.pivot(index="Date", columns="Symbol", values="Close")
        complete_symbols = pivot_test.columns[pivot_test.isna().sum() == 0].tolist()

        self.features = self.features[self.features["Symbol"].isin(complete_symbols)]

        return self.features

    def _get_edges(
        self,
        avg_rolling_corr: pd.DataFrame,
        symbols: list[str],
        device,
    ):
        # Vectorized approach using numpy
        corr_subset = avg_rolling_corr.loc[symbols, symbols]

        # Get upper triangle indices where correlation >= threshold
        mask = (corr_subset.values >= self.corr_threshold) & np.triu(
            np.ones_like(corr_subset.values, dtype=bool), k=1
        )

        # Get indices of edges
        src_indices, trgt_indices = np.where(mask)

        # Convert to edge index tensor directly
        edge_index = torch.tensor(
            [src_indices, trgt_indices], dtype=torch.int64, device=device
        )

        return edge_index

    def load_dataset(
        self,
        column: str = "Close",
        train_frac: float = 0.8,
        *,
        device,
    ):
        pivot = self.features.pivot(
            index="Date", columns="Symbol", values=column
        ).sort_index()

        rolling_corr = pivot.rolling(window=self.window_size).corr().dropna().abs()

        all_dates = rolling_corr.index.get_level_values(0).unique()
        usable_start_date = all_dates.min()
        usable_end_date = all_dates.max()

        for date in pd.bdate_range(
            usable_start_date, usable_end_date, freq=f"{self.window_size}B"
        ):
            date = pd.to_datetime(date)

            if date not in self.features["Date"].values:
                logger.warning(f"Skipping date {date} - no data available")
                continue

            logger.info(f"Processing date: {date}")

            features_subset = self.features[self.features["Date"] == date]

            if features_subset.empty:
                logger.warning(f"No features found for date {date}")
                continue

            all_symbols_at_date = features_subset["Symbol"].tolist()
            num_nodes = len(all_symbols_at_date)

            # Get next day data
            next_day_df = self.features[
                self.features["Date"] == date + pd.Timedelta(days=1)
            ].set_index("Symbol")

            pct_at_next_day = [
                next_day_df.loc[symbol, "PCT-1"] if symbol in next_day_df.index else 0.0
                for symbol in all_symbols_at_date
            ]

            pct_tensor = torch.tensor(pct_at_next_day, device=device)

            if not pct_at_next_day:
                logger.warning(f"No next day data for {date}")
                continue

            avg_corr_upto_date = rolling_corr.loc[:date].groupby(level=1).mean()
            edges = self._get_edges(avg_corr_upto_date, all_symbols_at_date, device)

            # ============================================================
            # USE VECTORIZED PAIR GENERATION (MUCH FASTER!)
            # ============================================================
            src_train, trgt_train, src_test, trgt_test = self._generate_pairs_vectorized(
                num_nodes, train_frac
            )

            # Convert numpy arrays to tensors
            src_idx_train_tensor = torch.tensor(src_train, device=device, dtype=torch.long)
            trgt_idx_train_tensor = torch.tensor(trgt_train, device=device, dtype=torch.long)
            src_idx_test_tensor = torch.tensor(src_test, device=device, dtype=torch.long)
            trgt_idx_test_tensor = torch.tensor(trgt_test, device=device, dtype=torch.long)

            features_tensor = torch.tensor(
                features_subset[ALL_FEATURES].to_numpy(), device=device
            )

            yield (
                features_tensor.float(),
                edges,
                src_idx_train_tensor,
                trgt_idx_train_tensor,
                src_idx_test_tensor,
                trgt_idx_test_tensor,
                pct_tensor.float(),
            )

    def _generate_pairs_vectorized(self, num_nodes: int, train_frac: float):
        """
        Generate train/test pairs using vectorized operations.
        ~10x faster than itertools.permutations for large graphs.
        """
        n = num_nodes
        total_pairs = n * (n - 1)

        # More efficient: use meshgrid and masking
        i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        
        # Mask out diagonal (i == j)
        mask = i_grid != j_grid
        
        i = i_grid[mask]
        j = j_grid[mask]

        # Randomly shuffle
        indices = np.random.permutation(total_pairs)
        split_point = int(total_pairs * train_frac)

        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        return (
            i[train_indices],
            j[train_indices],
            i[test_indices],
            j[test_indices],
        )

    def _get_cache_path(self):
        """Get path for cached graphs based on parameters."""
        cache_dir = Path(__file__).parent / "graph_cache"
        cache_dir.mkdir(exist_ok=True)
        filename = (
            f"graphs_w{self.window_size}_c{self.corr_threshold}_tf{self.train_frac}.pkl"
        )
        return cache_dir / filename

    def precompute_and_cache_graphs(self, train_frac: float = 0.8, device="cpu"):
        """
        Pre-compute ALL graphs once and save to disk.
        Call this ONCE before training.
        """
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"✓ Graphs already cached at {cache_path}")
            return

        print("Pre-computing all graphs (this may take a while)...")

        pivot = self.features.pivot(
            index="Date", columns="Symbol", values="Close"
        ).sort_index()

        rolling_corr = pivot.rolling(window=self.window_size).corr().dropna().abs()
        all_dates = rolling_corr.index.get_level_values(0).unique()
        usable_start_date = all_dates.min()
        usable_end_date = all_dates.max()

        cached_graphs = []

        for date in pd.bdate_range(
            usable_start_date, usable_end_date, freq=f"{self.window_size}B"
        ):
            date = pd.to_datetime(date)

            if date not in self.features["Date"].values:
                continue

            features_subset = self.features[self.features["Date"] == date]
            if features_subset.empty:
                continue

            all_symbols_at_date = features_subset["Symbol"].tolist()

            # Get next day PCT
            next_day_df = self.features[
                self.features["Date"] == date + pd.Timedelta(days=1)
            ].set_index("Symbol")

            pct_at_next_day = [
                next_day_df.loc[symbol, "PCT-1"] if symbol in next_day_df.index else 0.0
                for symbol in all_symbols_at_date
            ]

            if not pct_at_next_day:
                continue

            # Compute edges
            avg_corr_upto_date = rolling_corr.loc[:date].groupby(level=1).mean()
            edges_cpu = self._get_edges(avg_corr_upto_date, all_symbols_at_date, "cpu")

            # Generate train/test splits
            all_symbol_perm = list(permutations(all_symbols_at_date, 2))
            ticker_to_id_map = {
                symbol: idx for idx, symbol in enumerate(all_symbols_at_date)
            }

            train_split = sample(
                all_symbol_perm, int(len(all_symbol_perm) * train_frac)
            )
            test_split = [p for p in all_symbol_perm if p not in train_split]

            def make_indices(split):
                src = [ticker_to_id_map[s] for s, _ in split]
                trgt = [ticker_to_id_map[t] for _, t in split]
                return src, trgt

            src_train, trgt_train = make_indices(train_split)
            src_test, trgt_test = make_indices(test_split)

            # Store as numpy arrays (smaller than tensors)
            graph_data = {
                "date": date,
                "features": features_subset[ALL_FEATURES].to_numpy().astype(np.float32),
                "edges": edges_cpu.cpu().numpy(),
                "src_train": np.array(src_train, dtype=np.int64),
                "trgt_train": np.array(trgt_train, dtype=np.int64),
                "src_test": np.array(src_test, dtype=np.int64),
                "trgt_test": np.array(trgt_test, dtype=np.int64),
                "pct": np.array(pct_at_next_day, dtype=np.float32),
            }

            cached_graphs.append(graph_data)

            if len(cached_graphs) % 10 == 0:
                print(f"Cached {len(cached_graphs)} graphs...")

        # Save to disk
        with open(cache_path, "wb") as f:
            pickle.dump(cached_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ Cached {len(cached_graphs)} graphs to {cache_path}")
        print(f"  Cache size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")

    def load_dataset_from_cache(self, device):
        """
        Load pre-computed graphs from cache.
        MUCH faster than load_dataset()!
        """
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Graph cache not found at {cache_path}. "
                f"Run precompute_and_cache_graphs() first!"
            )

        print(f"Loading cached graphs from {cache_path}...")

        with open(cache_path, "rb") as f:
            cached_graphs = pickle.load(f)

        print(f"✓ Loaded {len(cached_graphs)} cached graphs")

        # Convert to tensors on-the-fly (fast)
        for graph_data in cached_graphs:
            yield (
                torch.tensor(
                    graph_data["features"], device=device, dtype=torch.float32
                ),
                torch.tensor(graph_data["edges"], device=device, dtype=torch.int64),
                torch.tensor(graph_data["src_train"], device=device, dtype=torch.int64),
                torch.tensor(
                    graph_data["trgt_train"], device=device, dtype=torch.int64
                ),
                torch.tensor(graph_data["src_test"], device=device, dtype=torch.int64),
                torch.tensor(graph_data["trgt_test"], device=device, dtype=torch.int64),
                torch.tensor(graph_data["pct"], device=device, dtype=torch.float32),
            )

    async def load_features_csv(self):
        path = Path(__file__).parent / "features.csv"
        if os.path.exists(path):
            self.features = pd.read_csv(path)
            self.features["Date"] = pd.to_datetime(
                self.features["Date"], format="mixed"
            )
            self.features.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.features.dropna(inplace=True)

            # NORMALIZE NUMERIC FEATURES (exclude one-hot encoded sectors AND PCT-1 target)
            numeric_features = [
                col
                for col in ALL_FEATURES
                if col not in ALL_SECTORS_FEATURES and col != "PCT-1"  # EXCLUDE TARGET!
            ]

            print(
                f"Normalizing {len(numeric_features)} numeric features (excluding PCT-1 target)"
            )
            print(
                f"NOT normalizing: PCT-1 (target), {len(ALL_SECTORS_FEATURES)} sector features"
            )
            print(
                f"Before normalization - Min: {self.features[numeric_features].min().min():.2f}, "
                f"Max: {self.features[numeric_features].max().max():.2f}"
            )

            self.scaler = StandardScaler()

            # FIT and TRANSFORM on all training data
            self.features[numeric_features] = self.scaler.fit_transform(
                self.features[numeric_features]
            )

            # Clip extreme outliers to ±5 standard deviations
            self.features[numeric_features] = self.features[numeric_features].clip(
                -5, 5
            )

            print(
                f"After normalization - Mean: {self.features[numeric_features].mean().mean():.4f}, "
                f"Std: {self.features[numeric_features].std().mean():.4f}"
            )
            print(
                f"After normalization - Min: {self.features[numeric_features].min().min():.2f}, "
                f"Max: {self.features[numeric_features].max().max():.2f}"
            )

            # SAVE SCALER for inference
            scaler_path = Path(__file__).parent / "feature_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Saved scaler to {scaler_path}")

            # SAVE CONFIG (which features to normalize)
            import json

            config_path = Path(__file__).parent / "normalization_config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "normalized_features": numeric_features,
                        "sector_features": ALL_SECTORS_FEATURES,
                        "target_feature": "PCT-1",
                        "clip_min": -5,
                        "clip_max": 5,
                    },
                    f,
                    indent=2,
                )
            print(f"✓ Saved normalization config to {config_path}")
        else:
            await self.async_gather_features()
