import os
import dotenv
import requests

dotenv.load_dotenv(override=True)

fmp_base_url = os.getenv("FMP_BASE_URL")


def call_fmp_api(function_name: str, **kwargs):
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY not found in environment variables.")
    response = requests.get(
        f"{fmp_base_url}/{function_name}",
        params={**kwargs, "apikey": fmp_api_key}
    )
    response.raise_for_status()
    return response.json()




def get_historical_data_for_company(ticker_symbol: str, start_date: str, end_date: str):
    return call_fmp_api(
        "historical-price-eod",
        symbol=ticker_symbol,
        from_=start_date,
        to=end_date
    )