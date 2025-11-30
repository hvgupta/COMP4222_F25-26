import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.logger import logger
from src.graph_builder import GraphManager
from src.feature_lists import ALL_FEATURES
from src.model import TwoTowerSAGE


class LinearRegressionBaseline:
    """
    Linear regression baseline that predicts target ticker's PCT-1 at day x+1
    using source ticker's features at day x and source ticker's PCT-1 at day x+1.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
    
    def prepare_data(self, GM: GraphManager, device='cpu'):
        """
        Extract training pairs from GraphManager.
        Returns: X_train, y_train, X_test, y_test
        """
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        logger.info("Preparing baseline dataset from GraphManager...")
        
        for (features_tensor, edges, src_idx_train, trgt_idx_train,
             src_idx_test, trgt_idx_test, pct_tensor) in GM.load_dataset(device=device):
            
            # Convert to numpy
            features = features_tensor.cpu().numpy()  # [N, F]
            pct_values = pct_tensor.cpu().numpy()  # [N]
            src_train = src_idx_train.cpu().numpy()
            trgt_train = trgt_idx_train.cpu().numpy()
            src_test = src_idx_test.cpu().numpy()
            trgt_test = trgt_idx_test.cpu().numpy()
            
            # Training pairs
            for src_i, trgt_i in zip(src_train, trgt_train):
                src_features = features[src_i]
                src_pct = pct_values[src_i]
                x = np.concatenate([src_features, [src_pct]])
                y = pct_values[trgt_i]
                
                X_train_list.append(x)
                y_train_list.append(y)
            
            # Test pairs
            for src_i, trgt_i in zip(src_test, trgt_test):
                src_features = features[src_i]
                src_pct = pct_values[src_i]
                x = np.concatenate([src_features, [src_pct]])
                y = pct_values[trgt_i]
                
                X_test_list.append(x)
                y_test_list.append(y)
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        
        logger.info(f"Prepared {len(X_train)} training samples, {len(X_test)} test samples")
        logger.info(f"Feature dimension: {X_train.shape[1]}")
        
        return X_train, y_train, X_test, y_test
    
    def train(self, X_train, y_train):
        """Fit the linear regression model."""
        logger.info("Training linear regression baseline...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred_train = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        
        logger.info(f"Training MSE: {train_mse:.6f}")
        logger.info(f"Training MAE: {train_mae:.6f}")
        
        return train_mse, train_mae
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Test MSE: {test_mse:.6f}")
        logger.info(f"Test MAE: {test_mae:.6f}")
        
        return {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'predictions': y_pred,
            'targets': y_test
        }


def load_gnn_model(model_path, device='cpu'):
    """Load the trained GNN model and its configuration."""
    model_dir = Path(model_path)
    
    # Verify paths exist
    history_path = model_dir / "history.json"
    model_weights_path = model_dir / "model.pth"
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
    
    # Load config from history.json
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    model_kwargs = config['model_kwargs']
    
    # Create model with same architecture
    model = TwoTowerSAGE(
        in_dim=len(ALL_FEATURES),
        hidden_dim=model_kwargs['hidden_dim'],
        out_dim=model_kwargs['out_dim'],
        dropout=model_kwargs['dropout'],
        embed_l2_reg=model_kwargs['embed_l2_reg'],
        normalize_embeddings=model_kwargs['normalize_embeddings']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    
    # Get best validation loss info
    val_losses = [h['val_loss'] for h in data['history']]
    best_loss = min(val_losses)
    best_epoch = data['history'][np.argmin(val_losses)]['epoch']
    
    logger.info(f"Loaded GNN model from {model_path}")
    logger.info(f"  Architecture: in={len(ALL_FEATURES)}, hidden={model_kwargs['hidden_dim']}, out={model_kwargs['out_dim']}")
    logger.info(f"  Best validation loss: {best_loss:.6f} at epoch {best_epoch}")
    
    return model, config, {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_loss': val_losses[-1],
        'all_losses': val_losses
    }


def evaluate_gnn_on_test(model, GM: GraphManager, device='cpu'):
    """Evaluate the loaded GNN model on test data."""
    model.eval()
    all_losses = []
    all_predictions = []
    all_targets = []
    
    logger.info("Evaluating GNN model on test data...")
    
    with torch.no_grad():
        for (features, edges, src_idx_train, trgt_idx_train,
             src_idx_test, trgt_idx_test, pct_tensor) in GM.load_dataset(device=device):
            
            # Test predictions - pass all required arguments to forward
            y_hat, E1, E2 = model(
                features, 
                edges, 
                src_idx_test, 
                trgt_idx_test, 
                pct_tensor
            )
            
            # Get targets
            targets = pct_tensor[trgt_idx_test]
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(y_hat, targets)
            
            all_losses.append(loss.item())
            all_predictions.extend(y_hat.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_test_loss = np.mean(all_losses)
    test_mae = mean_absolute_error(all_targets, all_predictions)
    
    logger.info(f"GNN Test MSE: {avg_test_loss:.6f}")
    logger.info(f"GNN Test MAE: {test_mae:.6f}")
    
    return {
        'test_mse': avg_test_loss,
        'test_mae': test_mae,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }


async def run_baseline_comparison(model_path, output_dir=None):
    """Run baseline and compare with loaded GNN model."""
    from src.graph_builder import WINDOW_SIZE, CORRELATION_THRESHOLD
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Model path: {model_path}")
    
    # Load GNN model
    logger.info("="*80)
    logger.info("Loading GNN Model...")
    logger.info("="*80)
    gnn_model, gnn_config, gnn_history = load_gnn_model(model_path, device=device)
    
    # Initialize GraphManager
    GM = GraphManager(WINDOW_SIZE, CORRELATION_THRESHOLD)
    await GM.load_features_csv()
    
    # Evaluate GNN on test data
    logger.info("\n" + "="*80)
    logger.info("Evaluating GNN on Test Data...")
    logger.info("="*80)
    gnn_results = evaluate_gnn_on_test(gnn_model, GM, device=device)
    
    # Train and evaluate baseline
    logger.info("\n" + "="*80)
    logger.info("Training Linear Regression Baseline...")
    logger.info("="*80)
    baseline = LinearRegressionBaseline()
    X_train, y_train, X_test, y_test = baseline.prepare_data(GM, device='cpu')
    baseline.train(X_train, y_train)
    baseline_results = baseline.evaluate(X_test, y_test)
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: GNN vs LINEAR REGRESSION BASELINE")
    logger.info("="*80)
    logger.info(f"\nGNN (Loaded Model from {model_path}):")
    logger.info(f"  Test MSE: {gnn_results['test_mse']:.6f}")
    logger.info(f"  Test MAE: {gnn_results['test_mae']:.6f}")
    logger.info(f"  Best Validation Loss (training): {gnn_history['best_loss']:.6f}")
    logger.info(f"  Number of predictions: {len(gnn_results['predictions'])}")
    
    logger.info(f"\nLinear Regression Baseline:")
    logger.info(f"  Test MSE: {baseline_results['test_mse']:.6f}")
    logger.info(f"  Test MAE: {baseline_results['test_mae']:.6f}")
    logger.info(f"  Number of predictions: {len(baseline_results['predictions'])}")
    
    # Calculate improvement
    improvement_mse = ((baseline_results['test_mse'] - gnn_results['test_mse']) / baseline_results['test_mse']) * 100
    improvement_mae = ((baseline_results['test_mae'] - gnn_results['test_mae']) / baseline_results['test_mae']) * 100
    
    logger.info(f"\nRelative Performance (MSE):")
    if improvement_mse > 0:
        logger.info(f"  ✓ GNN is {improvement_mse:.2f}% better than baseline")
    else:
        logger.info(f"  ✗ Baseline is {-improvement_mse:.2f}% better than GNN")
    
    logger.info(f"\nRelative Performance (MAE):")
    if improvement_mae > 0:
        logger.info(f"  ✓ GNN is {improvement_mae:.2f}% better than baseline")
    else:
        logger.info(f"  ✗ Baseline is {-improvement_mae:.2f}% better than GNN")
    
    logger.info(f"\nAbsolute Differences:")
    logger.info(f"  MSE: {abs(baseline_results['test_mse'] - gnn_results['test_mse']):.6f}")
    logger.info(f"  MAE: {abs(baseline_results['test_mae'] - gnn_results['test_mae']):.6f}")
    logger.info("="*80)
    
    # Save comparison results
    comparison = {
        'model_path': str(model_path),
        'gnn_test_mse': gnn_results['test_mse'],
        'gnn_test_mae': gnn_results['test_mae'],
        'gnn_best_val_loss': gnn_history['best_loss'],
        'gnn_best_epoch': gnn_history['best_epoch'],
        'baseline_mse': baseline_results['test_mse'],
        'baseline_mae': baseline_results['test_mae'],
        'improvement_mse_pct': improvement_mse,
        'improvement_mae_pct': improvement_mae,
        'num_test_samples_gnn': len(gnn_results['predictions']),
        'num_test_samples_baseline': len(X_test),
        'num_train_samples': len(X_train),
        'gnn_config': gnn_config
    }
    
    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / 'baseline_comparison.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path('baseline_comparison.json')
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"\n✓ Saved comparison results to {output_path}")
    
    return baseline, baseline_results, gnn_model, gnn_results, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare GNN model with Linear Regression baseline')
    parser.add_argument('--model-path', type=str, default='exp3/run_001',
                        help='Path to the directory containing model.pth and history.json (default: exp3/run_001)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save comparison results (default: current directory)')
    
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(run_baseline_comparison(args.model_path, args.output_dir))