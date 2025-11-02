from market_data_fetcher import (
    get_historical_eods_for_ticker,
    # get_company_financial_metrics,
)

import pdb
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "sp500_companies.csv"
save_path = current_dir / "one_year_company_info.csv"


def create_stock_price_dataframe(api_response: dict) -> pd.DataFrame:
    cur_info = []
    cur_symbol = api_response["symbol"]

    for daily_data in api_response["results"]:
        cur_date = datetime.fromtimestamp(daily_data["t"]/1000).strftime("%Y-%m-%d")
        cur_info.append(
            {
                "Date": cur_date,
                "Symbol": cur_symbol,
                "Open": daily_data["o"],
                "High": daily_data["h"],
                "Low": daily_data["l"],
                "Close": daily_data["c"],
                "Volume": daily_data["v"],
            }
        )

    return pd.DataFrame(columns=columns, data=cur_info)


if __name__ == "__main__":
    # pdb.set_trace()
    old_information = pd.read_csv(file_path)
    all_companies = set(old_information["Symbol"].tolist())

    exisiting_information = pd.DataFrame(columns=columns)
    exisiting_companies_list = set()
    if os.path.exists(save_path):
        exisiting_information = pd.read_csv(save_path)
        exisiting_companies_list = set(exisiting_information["Symbol"].tolist())

    for unseen_symbol in all_companies - exisiting_companies_list:
        try:

            historical_data = get_historical_eods_for_ticker(
                unseen_symbol, datetime(2024, 10, 1), datetime(2025, 10, 1)
            )
            if len(historical_data) == 0:
                print(f"Skipping {unseen_symbol} due to lack of data")
                continue

            cur_df = create_stock_price_dataframe(historical_data)
            exisiting_information = pd.concat(
                [exisiting_information, cur_df], ignore_index=True
            )

        except Exception as e:
            print(f"Error processing {unseen_symbol}: {e}")
            break

    exisiting_information.sort_values(by=["Date", "Symbol"], inplace=True)
    exisiting_information.to_csv("./one_year_company_info.csv", index=False)
    print(f"Added data for {unseen_symbol}")
