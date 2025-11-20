from src.logger import logger
from src.graph_builder import (
    GraphManager,
    WINDOW_SIZE,
    CORRELATION_THRESHOLD,
)
from src.model import TwoTowerSAGE
from src.feature_lists import ALL_FEATURES

import torch
import asyncio
import pandas as pd
import torch.optim as optim
from torch.nn import MSELoss, HuberLoss
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = HuberLoss(delta=0.05)  # robust loss for noisy finance data
model = TwoTowerSAGE().to(device=device)

logger.info(f"The device being used is: {device.type}")


async def train_model(
    window_size: int,
    corr_threshold: float,
    num_epoch: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 128,
):
    logger.info(
        f"Starting model training: epochs={num_epoch}, lr={learning_rate}, batch_size={batch_size}"
    )

    # Initialize GraphManager and gather features
    GM = GraphManager(window_size=window_size, corr_threshold=corr_threshold)
    logger.info("Loading company features...")
    await GM.load_features_csv()
    logger.info(f"Loaded features for {GM.features['Symbol'].nunique()} companies")

    train_data_points, test_data_points, date_info_map = GM.load_dataset(device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        all_dates: list[pd.Timestamp] = train_data_points["Date"].unique().tolist()
        for date in all_dates:
            date_specifc_dataset = train_data_points[train_data_points["Date"] == date]
            date_info = date_info_map[date]

            src_idx_tensor = torch.tensor(
                [
                    GM.ticker_to_id_map[ticker]
                    for ticker in date_specifc_dataset["src_node"].to_list()
                ],
                device=device,
            )

            trgt_idx_tensor = torch.tensor(
                [
                    GM.ticker_to_id_map[ticker]
                    for ticker in date_specifc_dataset["trgt_node"].to_list()
                ],
                device=device,
            )

            dataset = TensorDataset(src_idx_tensor, trgt_idx_tensor)

            for idx, batch in enumerate(
                DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
            ):
                src_idx, trgt_idx = batch

                optimizer.zero_grad()
                y_hat, E1, E2 = model(
                    date_info["node_features"],
                    date_info["edge_index"],
                    src_idx,
                    trgt_idx,
                    date_info["next_day_pct1"],
                )

                pred_loss = criterion(y_hat, date_info["next_day_pct1"][trgt_idx])
                # embedding L2 regularization
                reg_loss = model.embedding_regularization(E1, E2)

                loss = pred_loss + reg_loss

                loss.backward()
                optimizer.step()

                logger.info(
                    f"Current Date: {date.strftime("%Y-%m-%d")}, Current Batch Number: {idx}"
                )
        
        logger.info(f"Finished epoch number {epoch}")
    
    return model, test_data_points


def save_model(model: TwoTowerSAGE, filepath: str):
    """Save trained model to disk"""
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main():
    model, test_data_points = await train_model(WINDOW_SIZE, CORRELATION_THRESHOLD, num_epoch=30)
    if model is None:
        raise ValueError("Something is wrong here")

    save_model(model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())
