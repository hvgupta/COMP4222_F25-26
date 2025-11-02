from .market_data_fetcher import get_historical_eods_for_ticker, get_company_financial_metrics

import os
import pandas as pd
from datetime import datetime

if __name__  == "__main__":
    old_information = pd.read_csv("./sp500_companies.csv")
    all_companies = set(old_information["Symbol"].tolist())
    
    exisiting_information = pd.DataFrame()
    if os.path.exists("./one_year_company_info.csv"):
        exisiting_information = pd.read_csv("./one_year_company_info.csv")
        exisiting_companies_list = set(exisiting_information["Symbol"].tolist())
    
    for unseen_symbol in (all_companies - exisiting_companies_list):
        try:

            historical_data = get_historical_eods_for_ticker(
                unseen_symbol,
                datetime(2024, 10, 1),
                datetime(2025, 10, 1)
            )
            financial_metrics = get_company_financial_metrics(
                unseen_symbol
            )
            if len(historical_data) == 0 or len(financial_metrics) == 0:
                print(f"Skipping {unseen_symbol} due to lack of data")
                continue
            
            latest_close_price = historical_data[-1]["close"]
            latest_financials = financial_metrics[0]
            revenue = latest_financials.get("revenue", None)
            net_income = latest_financials.get("netIncome", None)
            
            new_row = pd.DataFrame([{
                "Symbol": unseen_symbol,
                "Latest Close Price": latest_close_price,
                "Revenue": revenue,
                "Net Income": net_income
            }])
            exisiting_information = pd.concat([exisiting_information, new_row], ignore_index=True)
            exisiting_information.to_csv("./one_year_company_info.csv", index=False)
            print(f"Added data for {unseen_symbol}")
        except Exception as e:
            print(f"Error processing {unseen_symbol}: {e}")