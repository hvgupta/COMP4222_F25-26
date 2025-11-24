from src.logger import logger
from src.graph_builder import GraphManager, WINDOW_SIZE, CORRELATION_THRESHOLD
from src.model import TwoTowerSAGE

import torch
import asyncio
import torch.optim as optim
from torch.nn import HuberLoss
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is: {device.type}")


async def train_model(
    GM: GraphManager,
    num_epoch: int = 100,
    learning_rate: float = 0.001,  # REDUCED - was causing NaN
    batch_size: int = 128,
):
    print(
        f"Starting model training: epochs={num_epoch}, lr={learning_rate}, batch_size={batch_size}"
    )
    print(f"Loaded features for {GM.features['Symbol'].nunique()} companies")

    # Initialize model
    # Initialize model
    model = TwoTowerSAGE().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HuberLoss(delta=0.05)

    for epoch in range(num_epoch):
        model.train()
        print(f"Starting epoch {epoch + 1}/{num_epoch}")

        epoch_loss = 0.0
        batch_count = 0
        graph_count = 0

        # Each iteration is one date/graph
        for (
            features_tensor,
            edges,
            src_idx_tensor,
            trgt_idx_tensor,
            pct_tensor,
        ) in GM.load_dataset(device=device):

            graph_count += 1

            # Validation checks
            if src_idx_tensor.numel() == 0 or trgt_idx_tensor.numel() == 0:
                print("No training pairs, skipping...")
                continue

            if features_tensor.numel() == 0 or edges.numel() == 0:
                print("Empty features or edges, skipping...")
                continue

            # Check for NaN/Inf in inputs
            if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
                print("NaN/Inf in features, skipping date...")
                continue

            if torch.isnan(pct_tensor).any() or torch.isinf(pct_tensor).any():
                print("NaN/Inf in pct_tensor, skipping date...")
                continue

            # LOG GRAPH STATISTICS (only in first epoch to avoid spam)
            if epoch == 0:
                num_nodes = features_tensor.shape[0]
                num_edges = edges.shape[1] if edges.numel() > 0 else 0
                num_pairs = len(src_idx_tensor)

                print(f"\n{'='*60}")
                print(f"Graph {graph_count}: {num_nodes} nodes, {num_edges} edges, {num_pairs} pairs")
                print(f"Features: min={features_tensor.min():.4f}, max={features_tensor.max():.4f}, mean={features_tensor.mean():.4f}")
                print(f"pct_tensor: min={pct_tensor.min():.4f}, max={pct_tensor.max():.4f}, mean={pct_tensor.mean():.4f}, std={pct_tensor.std():.4f}")
                print(f"{'='*60}\n")

            if pct_tensor.mean() == 0:
              print("skipping the graph with no pct change")
              continue


            # Create dataset of pairs
            dataset = TensorDataset(src_idx_tensor, trgt_idx_tensor)

            for idx, batch in enumerate(
                DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
            ):
                src_idx, trgt_idx = batch

                optimizer.zero_grad()

                try:
                    # Forward pass
                    y_hat, E1, E2 = model(
                        features_tensor,
                        edges,
                        src_idx,
                        trgt_idx,
                        pct_tensor,
                    )

                    # Ground truth
                    y_true = pct_tensor[trgt_idx]

                    # Check for NaN before loss
                    if torch.isnan(y_hat).any():
                        print("NaN in y_hat!")
                        print(f"E1 range: [{E1.min():.4f}, {E1.max():.4f}]")
                        print(f"E2 range: [{E2.min():.4f}, {E2.max():.4f}]")
                        print(f"pct_tensor range: [{pct_tensor.min():.4f}, {pct_tensor.max():.4f}]")
                        continue

                    # Losses
                    pred_loss = criterion(y_hat, y_true)
                    reg_loss = model.embedding_regularization(E1, E2)
                    loss = pred_loss + reg_loss

                    # Check loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN/Inf in loss! pred={pred_loss:.4f}, reg={reg_loss:.6f}")
                        continue

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                    if batch_count % 10 == 0:
                        print(
                            f"Epoch {epoch + 1}, Batch {batch_count}, Graph {graph_count}, "
                            f"Loss: {loss.item():.4f}, Pred: {pred_loss.item():.4f}, "
                            f"y_hat: [{y_hat.min():.4f}, {y_hat.max():.4f}], "
                            f"y_true: [{y_true.min():.4f}, {y_true.max():.4f}]"
                        )

                except Exception as e:
                    print(f"Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f}, Graphs processed: {graph_count}")
        else:
            print(f"Epoch {epoch + 1} had no valid batches!")

    return model



def save_model(model: TwoTowerSAGE, filepath: str):
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main():
    GM = GraphManager(WINDOW_SIZE, CORRELATION_THRESHOLD)
    model = await train_model(GM, num_epoch=100)
    save_model(model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())