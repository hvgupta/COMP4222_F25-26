from src.logger import logger
from src.graph_builder import GraphManager, WINDOW_SIZE, CORRELATION_THRESHOLD
from src.model import TwoTowerSAGE

import torch
import asyncio
import torch.optim as optim
from torch.nn import HuberLoss
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Initialize GraphManager
    GM = GraphManager(window_size=window_size, corr_threshold=corr_threshold)
    logger.info("Loading company features...")
    await GM.load_features_csv()
    logger.info(f"Loaded features for {GM.features['Symbol'].nunique()} companies")

    # Initialize model
    model = TwoTowerSAGE().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HuberLoss(delta=0.05)

    for epoch in range(num_epoch):
        model.train()
        logger.info(f"Starting epoch {epoch + 1}/{num_epoch}")
        
        epoch_loss = 0.0
        batch_count = 0

        # Each iteration is one date/graph
        for (
            features_tensor,
            edges,
            src_idx_tensor,
            trgt_idx_tensor,
            pct_tensor,
        ) in GM.load_dataset(device=device):
            
            # Validation checks
            if src_idx_tensor.numel() == 0 or trgt_idx_tensor.numel() == 0:
                logger.warning("No training pairs, skipping...")
                continue
            
            if features_tensor.numel() == 0 or edges.numel() == 0:
                logger.warning("Empty features or edges, skipping...")
                continue

            # Create dataset of pairs
            dataset = TensorDataset(src_idx_tensor, trgt_idx_tensor)

            for idx, batch in enumerate(
                DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
            ):
                logger.info(f"Processing batch {idx + 1} in epoch {epoch + 1}")
                src_idx, trgt_idx = batch

                optimizer.zero_grad()
                
                try:
                    y_hat, E1, E2 = model(
                        features_tensor,
                        edges,
                        src_idx,
                        trgt_idx,
                        pct_tensor,
                    )

                    # Ground truth
                    y_true = pct_tensor[trgt_idx]
                    
                    # Losses
                    pred_loss = criterion(y_hat, y_true)
                    reg_loss = model.embedding_regularization(E1, E2)
                    loss = pred_loss + reg_loss

                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1

                    if batch_count % 10 == 0:
                        logger.info(
                            f"Epoch {epoch + 1}, Batch {batch_count}, "
                            f"Loss: {loss.item():.4f}, Pred: {pred_loss.item():.4f}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error during training: {e}")
                    continue

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch + 1} had no valid batches!")

    return model


def save_model(model: TwoTowerSAGE, filepath: str):
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main():
    model = await train_model(WINDOW_SIZE, CORRELATION_THRESHOLD, num_epoch=30)
    save_model(model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())