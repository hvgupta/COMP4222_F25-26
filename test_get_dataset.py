import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_get_dataset():
    """Test the get_dataset function with existing features data."""
    
    # Load features
    features_path = Path(__file__).parent / "src" / "features_csv" / "features_test_output.csv"
    if not features_path.exists():
        print(f"Features file not found at {features_path}")
        return
    
    print("="*60)
    print("Loading Features Data")
    print("="*60)
    print(f"Loading from: {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"✓ Loaded {len(features_df)} rows")
    print(f"✓ Total columns: {len(features_df.columns)}")
    
    features_df["Date"] = pd.to_datetime(features_df["Date"], format='mixed')
    print(f"\nDate range: {features_df['Date'].min()} to {features_df['Date'].max()}")
    print(f"Unique symbols: {features_df['Symbol'].nunique()}")
    
    # Now let's try to import and test the GraphManager
    print("\n" + "="*60)
    print("Testing get_dataset Function")
    print("="*60)
    
    try:
        # We'll do a mock test with a simplified version
        print("\nCreating mock GraphManager with features data...")
        
        # Extract historical prices from features (we need Close and Volume at minimum)
        price_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close']
        # Check if Volume exists, otherwise skip it
        if 'Volume' in features_df.columns:
            price_cols.append('Volume')
        historical_prices = features_df[price_cols].copy()
        
        print(f"✓ Extracted historical prices: {len(historical_prices)} rows")
        
        # Now test the actual function
        print("\nImporting GraphManager...")
        from src.graph_builder import GraphManager
        
        gm = GraphManager()
        gm.features = features_df
        gm.historical_prices = historical_prices
        
        print("✓ GraphManager initialized")
        
        # Test with 2024 data
        print("\n" + "-"*60)
        print("Generating dataset for 2024...")
        print("-"*60)
        
        X, Y = gm.get_dataset(
            start_year=2024,
            end_year=2024,
            n_samples=100,
            price_column="Close"
        )
        
        print(f"\n✓ Dataset generation completed!")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        
        if not X.empty and not Y.empty:
            print(f"\nX columns ({len(X.columns)}): {list(X.columns)[:10]}...")
            print(f"Y columns: {list(Y.columns)}")
            
            print(f"\nFirst sample from X:")
            print(X.iloc[0].head(10))
            
            print(f"\nFirst sample from Y:")
            print(Y.iloc[0])
            
            print(f"\nX Data Summary:")
            print(f"  Rows: {len(X)}")
            print(f"  Cols: {len(X.columns)}")
            print(f"  Memory: {X.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            print(f"\nY Data Summary:")
            print(f"  Rows: {len(Y)}")
            print(f"  target_PCT_1 stats:\n{Y['target_PCT_1'].describe()}")
            
        else:
            print("⚠ Warning: Generated empty datasets")
            
    except ImportError as e:
        print(f"✗ Import Error (expected due to API calls): {e}")
        print("\nTo fully test, you need to:")
        print("1. Ensure all dependencies are installed")
        print("2. Set up proper environment variables if needed")
        print("3. Check network connectivity for API calls")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_dataset()
