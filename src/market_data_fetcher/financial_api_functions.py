from ..logger import logger

import requests
import pandas as pd
import yfinance as yf
from io import StringIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def fetch_sp500_companies():
    logger.info("Fetching S&P 500 company list from Wikipedia")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    sp500 = pd.read_html(StringIO(response.text))[0]

    logger.info("Successfully fetched S&P 500 company list")
    return sp500


def fetch_ticker_to_cik_map() -> dict[str, str]:
    logger.info("Fetching ticker to CIK mapping from SEC")

    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    ticker_to_cik_map = {
        info["ticker"]: str(info["cik_str"]).zfill(10)
        for info in response.json().values()
    }

    logger.info("Successfully fetched ticker to CIK mapping")
    return ticker_to_cik_map


def fetch_sec_concepts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")


def fetch_ticker_historical_prices(ticker_symbol: str, start_date: str, end_date: str):
    logger.info(
        f"Fetching historical prices for {ticker_symbol} from {start_date} to {end_date}"
    )

    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    logger.info(f"Successfully fetched historical prices for {ticker_symbol}")
    
    if data is None or data.empty:
        logger.warning(f"No historical price data found for {ticker_symbol}")
        raise ValueError(f"No historical price data found for {ticker_symbol}")
    
    return data


def extract_quarterly_data(facts: dict, metric_name: str, unit: str) -> pd.DataFrame:
    logger.info(f"Extracting quarterly data for {metric_name} in {unit}")

    if "us-gaap" not in facts:
        logger.warning("us-gaap data not found in facts")
        return pd.DataFrame()

    if metric_name not in facts["us-gaap"]:
        logger.warning(f"{metric_name} not found in us-gaap facts")
        return pd.DataFrame()

    if unit not in facts["us-gaap"][metric_name]["units"]:
        logger.warning(f"{unit} not found for {metric_name}")
        return pd.DataFrame()

    data = facts["us-gaap"][metric_name]["units"][unit]
    df = pd.DataFrame(data)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(by="end").reset_index(drop=True)

    logger.info(f"Successfully extracted quarterly data for {metric_name}")
    return df