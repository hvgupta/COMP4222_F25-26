from src.logger import logger
from src.graph_builder import (
    GraphManager,
    WINDOW_SIZE,
    CORRELATION_THRESHOLD,
)
from src.model import TwoTowerSAGE

import torch
import asyncio
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        logger.info(f"Starting epoch {epoch + 1}/{num_epoch}")

        # Load dataset generates data for each date
        for (
            features_tensor,
            edges,
            src_idx_tensor,
            trgt_idx_tensor,
            pct_tensor,
        ) in GM.load_dataset(device=device):
            # Skip if no training pairs were generated
            if len(src_idx_tensor) == 0 or len(trgt_idx_tensor) == 0:
                logger.warning("No training pairs for this date, skipping...")
                continue
            
            # Skip if tensors have mismatched sizes
            if len(src_idx_tensor) != len(trgt_idx_tensor):
                logger.warning("Mismatched tensor sizes, skipping...")
                continue

            dataset = TensorDataset(src_idx_tensor, trgt_idx_tensor)

            for idx, batch in enumerate(
                DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
            ):
                src_idx, trgt_idx = batch

                optimizer.zero_grad()
                y_hat, E1, E2 = model(
                    features_tensor,
                    edges,
                    src_idx,
                    trgt_idx,
                    pct_tensor,
                )

                pred_loss = criterion(y_hat, pct_tensor[trgt_idx])
                # embedding L2 regularization
                reg_loss = model.embedding_regularization(E1, E2)

                loss = pred_loss + reg_loss

                loss.backward()
                optimizer.step()

                if idx % 10 == 0:  # Log every 10 batches
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {idx}, Loss: {loss.item():.4f}"
                    )

        logger.info(f"Finished epoch {epoch + 1}")

    return model


def save_model(model: TwoTowerSAGE, filepath: str):
    """Save trained model to disk"""
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main():
    model = await train_model(WINDOW_SIZE, CORRELATION_THRESHOLD, num_epoch=30)
    if model is None:
        raise ValueError("Something is wrong here")

    save_model(model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())
