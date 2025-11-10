from logger import logger

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
        logger.error(f"Invalid quarter {quarter} for year {year}")
        raise ValueError("Quarter must be between 1 and 4")
    logger.debug(f"Quarter {quarter} for year {year}: start={start}, end={end}")
    return start, end


def _get_sum_of_prev_quarters(y_q_to_eps_map: dict, year: int):
    sum_prev = (
        y_q_to_eps_map.get((year, "Q1"), 0)
        + y_q_to_eps_map.get((year, "Q2"), 0)
        + y_q_to_eps_map.get((year, "Q3"), 0)
    )
    logger.debug(f"Sum of previous quarters for year {year}: {sum_prev}")
    return sum_prev


def clean_eps_table(eps_table: pd.DataFrame):
    logger.info(f"Starting to clean EPS table with {len(eps_table)} rows")
    filtered_eps = pd.DataFrame(columns=["start", "end", "eps", "fp"])
    y_q_to_eps_map = {}
    for i, row in eps_table.iterrows():
        row_dict = {
            "start": row["start"],
            "end": row["end"],
            "eps": row["val"],
            "fp": row["fp"],
        }
        check_quarter = determine_quarter(row["end"])
        if row["fp"] == "FY":
            logger.debug(
                f"Processing FY row at index {i}: end={row['end']}, val={row['val']}"
            )
            if check_quarter == 4:
                ammended_period = row["frame"]
                if pd.isna(ammended_period) or (
                    len(ammended_period) == 6
                ):  # only CY{YYYY} -> this just means that it is the full year eps
                    new_row_dict = row_dict.copy()
                    new_row_dict["fp"] = "Q4"
                    prev_sum = _get_sum_of_prev_quarters(
                        y_q_to_eps_map, row["end"].year
                    )
                    new_row_dict["eps"] = row["val"] - prev_sum
                    logger.info(
                        f"Calculated Q4 EPS for year {row['end'].year}: {new_row_dict['eps']} (FY val: {row['val']}, prev sum: {prev_sum})"
                    )
                    y_q_to_eps_map[(new_row_dict["end"].year, "Q4")] = new_row_dict
                else:
                    y = row["frame"][2:6]
                    q = int(row["frame"][-1])
                    if (int(y), f"Q{q}") in y_q_to_eps_map:
                        y_q_to_eps_map[(int(y), f"Q{q}")]["eps"] = row_dict["eps"]
                        logger.debug(
                            f"Updated existing EPS for {y} Q{q}: {row_dict['eps']}"
                        )
                    else:
                        start, end = get_start_and_end_of_quarter(int(y), q)
                        y_q_to_eps_map[(int(y), f"Q{q}")] = {
                            "start": start,
                            "end": end,
                            "eps": row_dict["eps"],
                            "fp": f"Q{q}",
                        }
                        logger.debug(
                            f"Added new EPS entry for {y} Q{q}: {row_dict['eps']}"
                        )
                    continue
            else:
                row_dict["fp"] = f"Q{check_quarter}"
                logger.warning(
                    f"FY row with non-Q4 end date {row['end']}, falling back to fp={row_dict['fp']}"
                )
        else:
            logger.debug(
                f"Processing non-FY row at index {i}: fp={row['fp']}, val={row['val']}"
            )

        y_q_to_eps_map[(row_dict["end"].year, row_dict["fp"])] = row_dict

    filtered_eps = pd.concat(
        [filtered_eps, pd.DataFrame(list(y_q_to_eps_map.values()))], ignore_index=True
    )
    logger.info(f"Cleaned EPS table: {len(filtered_eps)} rows after processing")
    return filtered_eps
