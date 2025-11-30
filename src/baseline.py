import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.logger import logger
from src.graph_builder import GraphManager
from src.feature_lists import ALL_FEATURES


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


def load_best_gnn_results(best_run_path="exp3/run_001"):
    """Load the best GNN model results from history.json"""
    history_path = Path(best_run_path) / "history.json"
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Get best validation loss
    val_losses = [h['val_loss'] for h in data['history']]
    best_loss = min(val_losses)
    best_epoch = data['history'][np.argmin(val_losses)]['epoch']
    
    return {
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_loss': val_losses[-1],
        'config': data['config'],
        'all_losses': val_losses
    }


async def run_baseline_comparison():
    """Run baseline and compare with best GNN model."""
    from src.graph_builder import WINDOW_SIZE, CORRELATION_THRESHOLD
    
    # Load best GNN results
    logger.info("="*80)
    logger.info("Loading best GNN model results...")
    logger.info("="*80)
    gnn_results = load_best_gnn_results("exp3/run_001")
    
    logger.info(f"GNN Best Validation Loss: {gnn_results['best_loss']:.6f} (Epoch {gnn_results['best_epoch']})")
    logger.info(f"GNN Final Loss: {gnn_results['final_loss']:.6f}")
    logger.info(f"GNN Config: {gnn_results['config']['model_kwargs']}")
    
    # Initialize GraphManager
    GM = GraphManager(WINDOW_SIZE, CORRELATION_THRESHOLD)
    await GM.load_features_csv()
    
    # Create and train baseline
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
    logger.info(f"\nGNN (Best Model - Run 001):")
    logger.info(f"  Best Validation Loss:  {gnn_results['best_loss']:.6f}")
    logger.info(f"  Final Validation Loss: {gnn_results['final_loss']:.6f}")
    
    logger.info(f"\nLinear Regression Baseline:")
    logger.info(f"  Test MSE: {baseline_results['test_mse']:.6f}")
    logger.info(f"  Test MAE: {baseline_results['test_mae']:.6f}")
    
    # Calculate improvement
    improvement_pct = ((baseline_results['test_mse'] - gnn_results['best_loss']) / baseline_results['test_mse']) * 100
    
    logger.info(f"\nRelative Performance:")
    if improvement_pct > 0:
        logger.info(f"  GNN is {improvement_pct:.2f}% better than baseline")
    else:
        logger.info(f"  Baseline is {-improvement_pct:.2f}% better than GNN")
    
    logger.info(f"\nAbsolute Difference: {abs(baseline_results['test_mse'] - gnn_results['best_loss']):.6f}")
    logger.info("="*80)
    
    # Save comparison results
    comparison = {
        'baseline_mse': baseline_results['test_mse'],
        'baseline_mae': baseline_results['test_mae'],
        'gnn_best_loss': gnn_results['best_loss'],
        'gnn_final_loss': gnn_results['final_loss'],
        'gnn_best_epoch': gnn_results['best_epoch'],
        'improvement_pct': improvement_pct,
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test)
    }
    
    with open('baseline_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info("\n✓ Saved comparison results to baseline_comparison.json")
    
    return baseline, baseline_results, gnn_results, comparison


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_baseline_comparison())