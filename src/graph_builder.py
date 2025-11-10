import ta
import pandas as pd
import networkx as nx

# ======= HYPER-PARAMETERS ======

WINDOW_SIZE = 5
"Number of Trading Days considered for correlation calculation"  # basically take the average over the window size

CORRELATION_THRESHOLD = 0.8
"Minimum correlation value to consider an edge between two tickers"

# ===============================


class GraphManager:
    def __init__(
        self,
        eod_data_csv: str,
        window_size=WINDOW_SIZE,
        corr_threshold=CORRELATION_THRESHOLD,
    ):
        self.historical_eod_data = pd.read_csv(eod_data_csv)
        self.window_size = window_size
        self.corr_threshold = corr_threshold
