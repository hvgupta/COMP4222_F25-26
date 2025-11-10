from wsgiref import headers
from ..logger import logger

import requests
import pandas as pd
import yfinance as yf
from io import StringIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_sp500_companies():
    logger.info("Fetching S&P 500 company list from Wikipedia")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    sp500 = pd.read_html(StringIO(response.text))[0]

    logger.info("Successfully fetched S&P 500 company list")
    return sp500


def get_ticker_to_cik_map() -> dict[str, str]:
    logger.info("Fetching ticker to CIK mapping from SEC")

    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url)
    ticker_to_cik_map = {
        info["ticker"]: str(info["cik_str"]).zfill(10)
        for info in response.json().values()
    }

    logger.info("Successfully fetched ticker to CIK mapping")
    return ticker_to_cik_map


def get_sec_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")


def get_ticker_historical_prices(ticker_symbol: str, start_date: str, end_date: str):
    logger.info(
        f"Fetching historical prices for {ticker_symbol} from {start_date} to {end_date}"
    )

    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    logger.info(f"Successfully fetched historical prices for {ticker_symbol}")
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


# def _call_api(endpoint: str, params: dict = {}):
#     response = requests.get(
#         endpoint,
#         params=params
#     )
#     response.raise_for_status()
#     return response.json()

# def _call_fmp_api(function_name: str, params: dict = {}):
#     fmp_api_key = os.getenv("FMP_API_KEY")
#     if not fmp_api_key:
#         raise ValueError("FMP_API_KEY not found in environment variables.")
#     _params = {**params, "apikey": fmp_api_key}
#     return _call_api(
#         f"{fmp_base_url}/{function_name}",
#         _params
#     )

# def _call_finage_api(function_name_and_param: str):
#     finage_api_key = os.getenv("FINAGE_API_KEY")
#     if not finage_api_key:
#         raise ValueError("FINAGE_API_KEY not found in environment variables.")

#     _params = {"apikey": finage_api_key}
#     return _call_api(
#         f"{finage_base_url}/{function_name_and_param}",
#         _params
#     )

# def get_company_financial_metrics(ticker_symbol: str):
#     return _call_fmp_api(
#         f"/stable/income-statement",
#         {"symbol": ticker_symbol}
#     )

# def get_historical_eods_for_ticker(ticker_symbol: str, from_date: datetime, to_date: datetime):
#     return _call_finage_api(
#         f"agg/stock/{ticker_symbol}/1/day/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
#     )
