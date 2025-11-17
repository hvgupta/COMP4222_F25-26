from src.logger import logger
from src.model import TwoTowerSAGE

import torch
from torch.nn import MSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = MSELoss()

def evaluate_model(model: TwoTowerSAGE, X_test, Y_test, edge_index, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained TwoTowerSAGE model
        X_test: Test feature tensor
        Y_test: Test target tensor
        edge_index: Edge index tensor
        device: Device (cpu or cuda)
    
    Returns:
        test_loss: MSE loss on test set
        predictions: Model predictions
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_test["target_PCT_1"].values, dtype=torch.float32, device=device)
        src_idx_tensor = torch.tensor(Y_test["source_node_idx"].values, dtype=torch.long, device=device)
        tgt_idx_tensor = torch.tensor(Y_test["target_node_idx"].values, dtype=torch.long, device=device)
        src_pct_tensor = torch.tensor(
            X_test["current_PCT_1"].values, dtype=torch.float32, device=device
        )

        y_hat, _, _ = model(
            X_tensor, edge_index, src_idx_tensor, tgt_idx_tensor, src_pct_tensor
        )
        test_loss = mse_loss(y_hat, Y_tensor)

    logger.info(f"Test Loss: {test_loss.item():.6f}")
    return test_loss.item(), y_hat.cpu().numpy()
