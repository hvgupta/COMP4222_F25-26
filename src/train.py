from .logger import logger
from .graph_builder import (
    GraphManager,
    START_YEAR,
    END_YEAR,
    WINDOW_SIZE,
    CORRELATION_THRESHOLD,
)
from .model import TwoTowerSAGE


import torch
import asyncio
import torch.optim as optim
from pandas import Timestamp
from torch.nn import MSELoss

mse_loss = MSELoss()
model = TwoTowerSAGE()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def train_model(
    window_size: int,
    corr_threshold: float,
    num_epoch: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """
    Train the TwoTowerSAGE model on stock price prediction.

    Args:
        window_size: Rolling window size for correlation
        corr_threshold: Minimum correlation for edges
        num_epoch: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    logger.info(
        f"Starting model training: epochs={num_epoch}, lr={learning_rate}, batch_size={batch_size}"
    )

    # Initialize GraphManager and gather features
    GM = GraphManager(window_size=window_size, corr_threshold=corr_threshold)

    logger.info("Gathering company features (this may take a while)...")
    await GM.async_gather_features()
    logger.info(f"Gathered features for {GM.features['Symbol'].nunique()} companies")

    # Build graph
    start_date = Timestamp(year=START_YEAR, month=1, day=1)
    end_date = Timestamp(year=END_YEAR, month=12, day=31)

    edges, _ = GM.build_graph(start_date, end_date, "Close")
    logger.info(f"Built graph with {len(edges)} edges")

    edge_index = GM.create_edge_index_to_tensor(edges, device)
    logger.info(f"Edge index tensor created on device: {device}")

    # Generate dataset
    X_train, Y_train = GM.get_dataset(START_YEAR, END_YEAR, n_samples=5000)

    if X_train.empty or Y_train.empty:
        logger.error("Failed to generate training dataset")
        return None

    logger.info(f"Generated training dataset with {len(X_train)} samples")

    # Convert to tensors
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(
        Y_train["target_PCT_1"].values, dtype=torch.float32, device=device
    )
    src_idx_tensor = torch.tensor(
        X_train["source_node_idx"].values, dtype=torch.long, device=device
    )
    tgt_idx_tensor = torch.tensor(
        Y_train["target_node_idx"].values, dtype=torch.long, device=device
    )
    src_pct_tensor = torch.tensor(
        X_train["current_PCT_1"].values, dtype=torch.float32, device=device
    )

    # Move model to device
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle indices
        shuffled_indices = torch.randperm(len(X_tensor))

        for batch_start in range(0, len(X_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(X_tensor))
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Get batch data
            X_batch = X_tensor[batch_indices]
            Y_batch = Y_tensor[batch_indices]
            src_idx_batch = src_idx_tensor[batch_indices]
            tgt_idx_batch = tgt_idx_tensor[batch_indices]
            src_pct_batch = src_pct_tensor[batch_indices]

            # Forward pass
            optimizer.zero_grad()
            y_hat, _, _ = model(
                X_batch, edge_index, src_idx_batch, tgt_idx_batch, src_pct_batch
            )

            # Compute loss
            loss = mse_loss(y_hat, Y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epoch} - Loss: {avg_epoch_loss:.6f}")

    logger.info("Training completed successfully")
    return model


def save_model(model: TwoTowerSAGE, filepath: str):
    """Save trained model to disk"""
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main(): 
    model = await train_model(WINDOW_SIZE, CORRELATION_THRESHOLD)
    if model is None:
        raise ValueError("Something is wrong here")
    
    save_model(model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())
