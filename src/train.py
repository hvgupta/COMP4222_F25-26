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
    """
    Train the TwoTowerSAGE model on stock price prediction.

    Args:
        window_size: Rolling window size for correlation
        corr_threshold: Minimum correlation for edges
        num_epoch: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: minibatch size for training
    """
    logger.info(f"Starting model training: epochs={num_epoch}, lr={learning_rate}, batch_size={batch_size}")

    # Initialize GraphManager and gather features
    GM = GraphManager(window_size=window_size, corr_threshold=corr_threshold)

    logger.info("Loading company features...")
    await GM.load_features_csv()
    logger.info(f"Loaded features for {GM.features['Symbol'].nunique()} companies")

    # Build graph
    start_date, end_date = GM.get_valid_date_range()
    if start_date is None or end_date is None:
        raise ValueError("Something is wrong")

    edges, _ = GM.build_graph(start_date, end_date, "Close")  # type: ignore
    logger.info(f"Built graph with {len(edges)} edges")

    edge_index = GM.conv_edge_index_to_tensor(edges, device)
    logger.info(f"Edge index tensor created on device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train, Y_train = GM.get_dataset(edges, start_date, end_date, n_samples=5000)
    # Convert full sampled dataset to tensors
    X_tensor = torch.tensor(
        X_train[ALL_FEATURES].values, dtype=torch.float32, device=device
    )
    Y_tensor = torch.tensor(
        Y_train["target_PCT_1"].values, dtype=torch.float32, device=device
    )*100
    src_idx_tensor = torch.tensor(
        X_train["source_node_idx"].values, dtype=torch.int64, device=device
    )
    tgt_idx_tensor = torch.tensor(
        Y_train["target_node_idx"].values, dtype=torch.int64, device=device
    )
    src_pct_tensor = torch.tensor(
        X_train["current_PCT_1"].values, dtype=torch.float32, device=device
    )
    
    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(num_epoch):
        epoch_loss = 0.0

        # Create DataLoader for minibatches
        dataset = TensorDataset(src_idx_tensor, tgt_idx_tensor, src_pct_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        num_batches = 0
        for batch in loader:
            src_idx_b, tgt_idx_b, src_pct_b, Yb = batch

            optimizer.zero_grad()
            y_hat, E1, E2 = model(
                X_tensor, edge_index, src_idx_b, tgt_idx_b, src_pct_b
            )

            # Compute loss (kept original scaling)
            pred_loss = criterion(y_hat, Yb)

            # embedding L2 regularization
            reg_loss = model.embedding_regularization(E1, E2)

            loss = pred_loss + reg_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / (num_batches or 1)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epoch} - Avg Loss: {avg_epoch_loss:.6f}")

    logger.info("Training completed successfully")
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
