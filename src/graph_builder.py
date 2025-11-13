from .company_feature_functions import *
from .feature_lists import *

import pandas as pd
from pandas import Timestamp


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

    def gather_features(self):
        start_date = Timestamp(year=self.start_year - 1, month=12, day=1)
        end_date = Timestamp(year=self.end_year, month=12, day=31)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        for i, row in self.company_df.iterrows():

            if i == 5:
                break

            ticker = row["Symbol"]
            prices = fetch_ticker_historical_prices(
                ticker, start_date_str, end_date_str
            )

            self.historical_prices.loc[:, pd.IndexSlice[:, ticker]] = prices  # type: ignore

            company_concept = fetch_sec_concepts(TICKER_TO_CIK_MAP[ticker])

            company_features = pd.DataFrame(columns=ALL_FEATURES)

            company_price_features = get_historical_price_features(ticker, prices)
            # company_price_features = company_price_features[]
            company_features[["Date"] + HISTORICAL_DATA_FEATURES] = (
                company_price_features
            )

            company_PE_ratio_df = get_PE_ratio_data(
                ticker, prices, company_concept["facts"], self.start_year, self.end_year
            )
            company_features = self._merge_into_company_features(
                company_features, company_PE_ratio_df, PE_FEATURES
            )
            try:
                company_PB_ratio_df = get_PB_ratio_data(
                    ticker, prices, company_concept["facts"], self.start_year, self.end_year
                )
            except SKIPException as e:
                logger.warning(f"Skipping {ticker} due to missing data: {e}")
                continue
            
            company_features = self._merge_into_company_features(
                company_features, company_PB_ratio_df, PB_FEATURES
            )

            company_ROA_df = get_roa_data(
                ticker, company_concept["facts"], self.start_year, self.end_year
            )
            company_features = self._merge_into_company_features(
                company_features, company_ROA_df, ROA_FEATURES
            )

            company_current_ratio_df = get_current_ratio_data(
                ticker, company_concept["facts"], self.start_year, self.end_year
            )
            company_features = self._merge_into_company_features(
                company_features, company_current_ratio_df, CURRENT_FEATURES
            )

            company_features["Symbol"] = ticker
            sector_one_hot = get_one_hot_sector(row["GICS Sector"], ALL_GICS_SECTORS)
            company_features[sector_one_hot.index] = sector_one_hot.values


            self.features = pd.concat([self.features, company_features], ignore_index=True)

        return self.features
