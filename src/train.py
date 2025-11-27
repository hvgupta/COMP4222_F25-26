from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


@dataclass
class TrainingSummary:
    model: TwoTowerSAGE
    config: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def best_epoch(self) -> Optional[Dict[str, Any]]:
        """Returns the epoch with lowest validation loss."""
        if not self.history:
            return None
        return min(self.history, key=lambda entry: entry.get("val_loss", float("inf")))

    @property
    def best_train_epoch(self) -> Optional[Dict[str, Any]]:
        """Returns the epoch with lowest training loss."""
        if not self.history:
            return None
        return min(self.history, key=lambda entry: entry["train_loss"])


async def train_model(
    GM: GraphManager,
    num_epoch: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    model_kwargs: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,  # NEW: Enable caching
) -> TrainingSummary:
    print(
        f"Starting model training: epochs={num_epoch}, lr={learning_rate}, batch_size={batch_size}"
    )
    
    if GM.features.empty:
        await GM.load_features_csv()
    
    # PRE-COMPUTE GRAPHS ONCE (if not already cached)
    if use_cache:
        GM.precompute_and_cache_graphs(train_frac=0.8, device='cpu')
    
    model_params = model_kwargs or {}
    model = TwoTowerSAGE(**model_params).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HuberLoss(delta=0.05)
    history: List[Dict[str, Any]] = []

    for epoch in range(num_epoch):
        model.train()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epoch}")
        print(f"{'='*80}")

        train_epoch_loss = 0.0
        train_batch_count = 0
        train_graph_count = 0
        
        val_epoch_loss = 0.0
        val_batch_count = 0
        val_graph_count = 0

        # USE CACHED GRAPHS (MUCH FASTER!)
        dataset_loader = (
            GM.load_dataset_from_cache(device=device) 
            if use_cache 
            else GM.load_dataset(device=device)
        )

        for (
            features_tensor,
            edges,
            src_idx_train_tensor,
            trgt_idx_train_tensor,
            src_idx_test_tensor,
            trgt_idx_test_tensor,
            pct_tensor,
        ) in dataset_loader:

            train_graph_count += 1

            # ============================================================
            # TRAINING on train pairs
            # ============================================================
            if src_idx_train_tensor.numel() == 0 or trgt_idx_train_tensor.numel() == 0:
                print("No training pairs, skipping graph...")
                continue

            if features_tensor.numel() == 0 or edges.numel() == 0:
                print("Empty features or edges, skipping graph...")
                continue

            # Check for NaN/Inf in inputs
            if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
                print("NaN/Inf in features, skipping graph...")
                continue

            if torch.isnan(pct_tensor).any() or torch.isinf(pct_tensor).any():
                print("NaN/Inf in pct_tensor, skipping graph...")
                continue

            # LOG GRAPH STATISTICS (only in first epoch)
            if epoch == 0:
                num_nodes = features_tensor.shape[0]
                num_edges = edges.shape[1] if edges.numel() > 0 else 0
                num_train_pairs = len(src_idx_train_tensor)
                num_test_pairs = len(src_idx_test_tensor)

                print(f"\nGraph {train_graph_count}:")
                print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
                print(f"  Train pairs: {num_train_pairs}, Test pairs: {num_test_pairs}")
                print(f"  Features range: [{features_tensor.min():.4f}, {features_tensor.max():.4f}]")
                print(f"  PCT range: [{pct_tensor.min():.4f}, {pct_tensor.max():.4f}]")

            if pct_tensor.mean() == 0:
                print("Skipping graph with no pct change")
                continue

            # Create training dataset of pairs
            train_dataset = TensorDataset(src_idx_train_tensor, trgt_idx_train_tensor)

            for batch in DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False):
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

                    # Check for NaN
                    if torch.isnan(y_hat).any():
                        print("NaN in y_hat! Skipping batch...")
                        continue

                    # Losses
                    pred_loss = criterion(y_hat, y_true)
                    reg_loss = model.embedding_regularization(E1, E2)
                    loss = pred_loss + reg_loss

                    # Check loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN/Inf in loss! Skipping batch...")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_epoch_loss += loss.item()
                    train_batch_count += 1

                    if train_batch_count % 10 == 0:
                        print(
                            f"[TRAIN] Batch {train_batch_count}, Graph {train_graph_count}, "
                            f"Loss: {loss.item():.4f}, Pred: {pred_loss.item():.4f}"
                        )

                except Exception as e:
                    print(f"Error during training: {e}")
                    continue

            # ============================================================
            # VALIDATION on test pairs (if available)
            # ============================================================
            if src_idx_test_tensor.numel() > 0 and trgt_idx_test_tensor.numel() > 0:
                model.eval()  # Switch to evaluation mode
                val_graph_count += 1

                test_dataset = TensorDataset(src_idx_test_tensor, trgt_idx_test_tensor)

                with torch.no_grad():
                    for batch in DataLoader(test_dataset, batch_size, shuffle=False):
                        src_idx, trgt_idx = batch

                        try:
                            # Forward pass (no gradient)
                            y_hat, E1, E2 = model(
                                features_tensor,
                                edges,
                                src_idx,
                                trgt_idx,
                                pct_tensor,
                            )

                            y_true = pct_tensor[trgt_idx]

                            if torch.isnan(y_hat).any():
                                continue

                            # Compute validation loss
                            pred_loss = criterion(y_hat, y_true)
                            reg_loss = model.embedding_regularization(E1, E2)
                            loss = pred_loss + reg_loss

                            if torch.isnan(loss) or torch.isinf(loss):
                                continue

                            val_epoch_loss += loss.item()
                            val_batch_count += 1

                        except Exception as e:
                            print(f"Error during validation: {e}")
                            continue

                model.train()  # Switch back to training mode

        # ============================================================
        # EPOCH SUMMARY
        # ============================================================
        avg_train_loss = train_epoch_loss / train_batch_count if train_batch_count > 0 else float("inf")
        avg_val_loss = val_epoch_loss / val_batch_count if val_batch_count > 0 else None

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_batches": train_batch_count,
            "train_graphs": train_graph_count,
        }

        if avg_val_loss is not None:
            epoch_summary.update({
                "val_loss": avg_val_loss,
                "val_batches": val_batch_count,
                "val_graphs": val_graph_count,
            })

        history.append(epoch_summary)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f} ({train_batch_count} batches, {train_graph_count} graphs)")
        if avg_val_loss is not None:
            print(f"  Val Loss:   {avg_val_loss:.6f} ({val_batch_count} batches, {val_graph_count} graphs)")
        else:
            print(f"  Val Loss:   N/A (no test data)")
        print(f"{'='*80}\n")

    summary = TrainingSummary(
        model=model,
        config={
            "num_epoch": num_epoch,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_kwargs": model_params,
            "window_size": GM.window_size,
            "corr_threshold": GM.corr_threshold,
        },
        history=history,
    )

    # Print best epoch info
    if summary.best_epoch:
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Epoch (by validation loss): {summary.best_epoch['epoch']}")
        print(f"  Train Loss: {summary.best_epoch['train_loss']:.6f}")
        if 'val_loss' in summary.best_epoch:
            print(f"  Val Loss:   {summary.best_epoch['val_loss']:.6f}")
        print(f"{'='*80}\n")

    return summary


def save_model(model: TwoTowerSAGE, filepath: str):
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


async def main():
    GM = GraphManager(WINDOW_SIZE, CORRELATION_THRESHOLD)
    await GM.load_features_csv()
    summary = await train_model(GM, num_epoch=100)
    save_model(summary.model, "./corr_model.pth")


if __name__ == "__main__":
    asyncio.run(main())