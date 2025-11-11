from .logger import logger

import pandas as pd
from pandas import Timestamp


def determine_quarter(end_date: Timestamp):
    quarter = end_date.quarter
    logger.debug(f"Determined quarter {quarter} for end_date {end_date}")
    return quarter


def get_start_and_end_of_quarter(year: int, quarter: int):
    if quarter == 1:
        start, end = Timestamp(year=year, month=1, day=1), Timestamp(
            year=year, month=3, day=31
        )
    elif quarter == 2:
        start, end = Timestamp(year=year, month=4, day=1), Timestamp(
            year=year, month=6, day=30
        )
    elif quarter == 3:
        start, end = Timestamp(year=year, month=7, day=1), Timestamp(
            year=year, month=9, day=30
        )
    elif quarter == 4:
        start, end = Timestamp(year=year, month=10, day=1), Timestamp(
            year=year, month=12, day=31
        )
    else:
        start, end = Timestamp(year=year, month=1, day=1), Timestamp(
            year=year, month=12, day=31
        )
    logger.debug(f"Quarter {quarter} for year {year}: start={start}, end={end}")
    return start, end


def _get_sum_of_prev_quarters(y_q_to_eps_map: dict, year: int):
    sum_prev = (
        y_q_to_eps_map.get((year, "Q1"), {}).get("eps", 0)
        + y_q_to_eps_map.get((year, "Q2"), {}).get("eps", 0)
        + y_q_to_eps_map.get((year, "Q3"), {}).get("eps", 0)
    )
    logger.debug(f"Sum of previous quarters for year {year}: {sum_prev}")
    return sum_prev


def _eps_get_start_end_filter(eps_table: pd.DataFrame, year: int, quarter: str):
    start, end = get_start_and_end_of_quarter(
        year, int(quarter[1]) if quarter[1].isdigit() else 0
    )
    return (eps_table["start"] >= start) & (eps_table["end"] <= end)


def clean_eps_table(eps_table: pd.DataFrame, start_year: int, end_year: int):
    logger.info(f"Starting to clean EPS table with {len(eps_table)} rows")
    filtered_eps = pd.DataFrame(columns=["start", "end", "eps", "fp"])
    y_q_to_eps_map = {}
    for year in range(start_year, end_year + 1):
        for quarter in ["Q1", "Q2", "Q3", "Q4", "FY"]:
            subset = eps_table[
                (eps_table["end"].dt.year == year)
                & (eps_table["fp"] == quarter)
                & _eps_get_start_end_filter(eps_table, year, quarter)
            ].sort_values("filed", ascending=False)
            """Gets the subset of values where the year and quarter match up and the start and end date are correct according to the quarter"""

            if subset.empty:
                logger.debug(f"No data for year {year} quarter {quarter}")
                continue

            y_q_eps_table = subset.iloc[0]  # safe: subset has at least one row
            y_q_to_eps_map[(year, quarter)] = {
                "start": y_q_eps_table["start"],
                "end": y_q_eps_table["end"],
                "eps": y_q_eps_table["val"],
                "fp": quarter,
            }

        # Q4 fallback: compute from FY if Q4 missing (do this per-year)
        if (year, "Q4") not in y_q_to_eps_map and (year, "FY") in y_q_to_eps_map:
            start, end = get_start_and_end_of_quarter(year, 4)
            y_q_to_eps_map[(year, "Q4")] = {
                "start": start,
                "end": end,
                "eps": y_q_to_eps_map[(year, "FY")]["eps"]
                - _get_sum_of_prev_quarters(y_q_to_eps_map, year),
                "fp": "Q4",
            }

    filtered_eps = pd.concat(
        [filtered_eps, pd.DataFrame(list(y_q_to_eps_map.values()))], ignore_index=True
    )
    logger.info(f"Cleaned EPS table: {len(filtered_eps)} rows after processing")
    return filtered_eps


def clean_instance_tables(
    instance_table: pd.DataFrame, start_year: int, end_year: int, quantity_name: str
):
    filtered_instance = pd.DataFrame(columns=["start", "end", quantity_name, "fp"])
    y_q_to_equity_list = []
    for year in range(start_year, end_year + 1):
        for month in [3, 6, 9, 12]:
            subset = instance_table[
                (instance_table["end"].dt.year == year)
                & (instance_table["end"].dt.month == month)
            ].sort_values("filed", ascending=False)

            if subset.empty:
                continue

            latest_row = subset.iloc[0]
            current_quarter = month // 3
            start, end = get_start_and_end_of_quarter(year, (month // 3))

            if (
                month == 12
            ):  # since the stockholder equity is for a particular instance, so Q4 is the same as FY
                y_q_to_equity_list.append(
                    {
                        "start": start,
                        "end": end,
                        quantity_name: latest_row["val"],
                        "fp": "Q4",
                    }
                )
                y_q_to_equity_list.append(
                    {
                        "start": Timestamp(year=year, month=1, day=1),
                        "end": end,
                        quantity_name: latest_row["val"],
                        "fp": "FY",
                    }
                )
            else:
                y_q_to_equity_list.append(
                    {
                        "start": start,
                        "end": end,
                        quantity_name: latest_row["val"],
                        "fp": f"Q{current_quarter}",
                    }
                )

    filtered_instance = pd.DataFrame.from_records(y_q_to_equity_list)

    for year in range(start_year, end_year + 1):
        for quarter in ["Q1", "Q2", "Q3", "Q4", "FY"]:
            if not filtered_instance.query(
                f"end.dt.year == {year} and fp == '{quarter}'"
            ).empty:
                continue
            start, end = get_start_and_end_of_quarter(
                year, int(quarter[1]) if quarter[1].isdigit() else 0
            )
            filtered_instance.loc[len(filtered_instance)] = {  # type: ignore
                "start": start,
                "end": end,
                quantity_name: None,
                "fp": quarter,
            }

    filtered_instance.sort_values(
        ["end", "start"], ascending=[True, False], inplace=True
    )
    filtered_instance.reset_index(drop=True, inplace=True)

    filtered_instance[quantity_name] = filtered_instance[quantity_name].bfill()
    filtered_instance.dropna(inplace=True)

    return filtered_instance
