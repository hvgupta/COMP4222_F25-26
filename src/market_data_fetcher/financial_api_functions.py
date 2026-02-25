from ..logger import logger

import aiohttp
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from typing import Dict

HEADERS = {"User-Agent": "Mozilla/5.0 (Company info@company.com)"}


def fetch_sp500_companies() -> pd.DataFrame:
    logger.info("Fetching S&P 500 company list from Wikipedia (async)")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    sp500 = pd.read_html(StringIO(response.text))[0]

    logger.info("Successfully fetched S&P 500 company list")
    return sp500


def fetch_ticker_to_cik_map() -> Dict[str, str]:
    logger.info("Fetching ticker to CIK mapping from SEC (async)")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    ticker_to_cik_map = {
        info["ticker"]: str(info["cik_str"]).zfill(10)
        for info in response.json().values()
    }

    logger.info("Successfully fetched ticker to CIK mapping")
    return ticker_to_cik_map


async def fetch_sec_concepts(cik: str) -> dict:
    logger.info(f"Fetching SEC concepts for CIK={cik} (async)")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to fetch data, code: {resp.status}, reason: {resp.reason or (await resp.text())}")


async def fetch_ticker_historical_prices(
    ticker_symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    yfinance is synchronous; run the download call in a thread to avoid blocking the event loop.
    """
    logger.info(
        f"Fetching historical prices for {ticker_symbol} from {start_date} to {end_date} (async)"
    )

    data = yf.download(
        ticker_symbol, start=start_date, end=end_date, multi_level_index=False
    )

    logger.info(f"Successfully fetched historical prices for {ticker_symbol}")

    if data is None or data.empty:
        logger.warning(f"No historical price data found for {ticker_symbol}")
        raise ValueError(f"No historical price data found for {ticker_symbol}")

    data.reset_index(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])

    return data