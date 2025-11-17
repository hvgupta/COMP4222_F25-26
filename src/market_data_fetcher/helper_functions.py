from src.logger import logger

import pandas as pd
from pandas import Timestamp


def determine_quarter(end_date: Timestamp):
    quarter = end_date.quarter
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
    return start, end


def _get_frame_data(eps_table: pd.DataFrame, year: int, quarter: str):
    subset = eps_table[
        (eps_table["end"].dt.year == year)
        & (eps_table["frame"] == f"CY{year}{quarter}")
    ].sort_values("filed", ascending=False)

    if subset.empty:
        logger.debug(f"No data for year {year} quarter {quarter}")
        return None

    return subset.iloc[0]


def _compute_missing_quarter_from_fy(
    y_q_to_quantity_map: dict, year: int, quantity_name: str, fy_val: float
) -> dict:

    # collect present quarters
    present = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        v = y_q_to_quantity_map.get((year, q), {})
        val = v.get(quantity_name) if isinstance(v, dict) else None
        try:
            val_num = float(val) if val is not None else None
        except Exception:
            val_num = None
        if val_num is not None:
            present[q] = val_num

    if len(present) == 4:
        logger.debug(f"All quarters present for {year}, no imputation needed")
        return y_q_to_quantity_map

    missing_qs = [q for q in ("Q1", "Q2", "Q3", "Q4") if q not in present]
    if len(missing_qs) != 1:
        logger.debug(
            f"{year} has {len(missing_qs)} missing quarters; need exactly 1 to impute"
        )
        return y_q_to_quantity_map

    missing_q = missing_qs[0]
    sum_three = sum(present.values())

    imputed_val = fy_val - sum_three
    start, end = get_start_and_end_of_quarter(year, int(missing_q[1]))
    y_q_to_quantity_map[(year, missing_q)] = {
        "start": start,
        "end": end,
        quantity_name: imputed_val,
        "fp": missing_q,
    }
    logger.info(
        f"Computed {missing_q} for {year}: {quantity_name}={imputed_val} "
        f"(FY {fy_val} - sum_other_three {sum_three})"
    )
    return y_q_to_quantity_map


def clean_period_table(
    eps_table: pd.DataFrame, start_year: int, end_year: int, quantity_name: str
):
    logger.info(f"Starting to clean EPS table with {len(eps_table)} rows")
    filtered_period_table = pd.DataFrame(columns=["start", "end", quantity_name, "fp"])
    y_q_to_quantity_map = {}
    for year in range(start_year, end_year + 1):
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:

            y_q_eps_table = _get_frame_data(eps_table, year, quarter)
            if y_q_eps_table is None:
                continue

            start, end = get_start_and_end_of_quarter(year, int(quarter[1]))

            y_q_to_quantity_map[(year, quarter)] = {
                "start": start,
                "end": end,
                quantity_name: y_q_eps_table["val"],
                "fp": quarter,
            }

        y_q_eps_table = _get_frame_data(eps_table, year, "")
        if y_q_eps_table is None:
            logger.warning(f"No FY data for year {year}, cannot impute missing quarters")
            continue

        y_q_to_quantity_map = _compute_missing_quarter_from_fy(
            y_q_to_quantity_map, year, quantity_name, y_q_eps_table["val"]
        )

    filtered_period_table = pd.concat(
        [filtered_period_table, pd.DataFrame(list(y_q_to_quantity_map.values()))],
        ignore_index=True,
    )
    logger.info(
        f"Cleaned period table: {len(filtered_period_table)} rows after processing"
    )

    filtered_period_table.sort_values(
        ["end", "start"], ascending=[True, False], inplace=True
    )
    filtered_period_table.reset_index(drop=True, inplace=True)

    return filtered_period_table


def clean_instance_tables(
    instance_table: pd.DataFrame, start_year: int, end_year: int, quantity_name: str
):
    filtered_instance = pd.DataFrame(columns=["start", "end", quantity_name, "fp"])
    y_q_to_equity_list = []
    for year in range(start_year, end_year + 1):
        for quarter in ["Q1I", "Q2I", "Q3I", "Q4I"]:
            subset = instance_table[
                (instance_table["end"].dt.year == year)
                & (instance_table["frame"] == f"CY{year}{quarter}")
            ].sort_values("filed", ascending=False)

            if subset.empty:
                continue

            latest_row = subset.iloc[0]
            start, end = get_start_and_end_of_quarter(year, int(quarter[1]))
            y_q_to_equity_list.append(
                {
                    "start": start,
                    "end": end,
                    quantity_name: latest_row["val"],
                    "fp": quarter[:-1],
                }
            )

    filtered_instance = pd.DataFrame.from_records(y_q_to_equity_list)
    if filtered_instance.empty:
        logger.warning("No data found in instance table after filtering")
        raise ValueError("No data found in instance table after filtering")

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
