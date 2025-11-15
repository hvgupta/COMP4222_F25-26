import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from itertools import combinations

def test_get_dataset_standalone():
    """
    Standalone test of the get_dataset logic without importing full GraphManager.
    This replicates the get_dataset function with the actual features data.
    """
    
    # Load features
    features_path = Path(__file__).parent / "src" / "features_csv" / "features_test_output.csv"
    print("="*70)
    print("STANDALONE GET_DATASET TEST")
    print("="*70)
    print(f"\nLoading features from: {features_path}")
    
    features_df = pd.read_csv(features_path)
    features_df["Date"] = pd.to_datetime(features_df["Date"], format='mixed')
    
    print(f"✓ Loaded {len(features_df)} rows")
    print(f"✓ Date range: {features_df['Date'].min().date()} to {features_df['Date'].max().date()}")
    print(f"✓ Unique symbols: {features_df['Symbol'].nunique()}")
    
    # Create historical prices dataframe
    price_cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close']
    if 'Volume' in features_df.columns:
        price_cols.append('Volume')
    
    historical_prices = features_df[price_cols].copy()
    
    # Simulate get_dataset for year 2024
    start_year = 2024
    end_year = 2024
    n_samples = 100
    price_column = "Close"
    corr_threshold = 0.8
    window_size = 5
    
    print(f"\n" + "="*70)
    print(f"Generating Dataset: {start_year}-{end_year} | Samples: {n_samples}")
    print("="*70)
    
    # STEP 1: Build correlation graph
    print(f"\nStep 1: Building correlation graph...")
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)
    
    df = historical_prices.copy()
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    
    pivot = df.pivot(index="Date", columns="Symbol", values=price_column).sort_index()
    symbols = pivot.columns.tolist()
    
    print(f"  Date range: {pivot.index.min().date()} to {pivot.index.max().date()}")
    print(f"  Symbols in range: {len(symbols)}")
    print(f"  Trading days: {len(pivot)}")
    
    # Compute rolling correlations
    rolling_corr = pivot.rolling(window_size).corr()
    avg_corr_matrix = rolling_corr.groupby(level=1).mean()
    
    # Build edge list
    edges = []
    for i, j in combinations(symbols, 2):
        corr_val = avg_corr_matrix.loc[i, j]
        if pd.notna(corr_val) and corr_val >= corr_threshold:
            edges.append((i, j, float(corr_val)))
    
    print(f"  ✓ Edges found (corr >= {corr_threshold}): {len(edges)}")
    
    if not edges:
        print("  ✗ No edges found. Cannot generate dataset.")
        return
    
    # STEP 2: Create ticker to index mapping
    print(f"\nStep 2: Creating node index mapping...")
    unique_symbols = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(unique_symbols)}
    print(f"  ✓ Unique nodes: {len(unique_symbols)}")
    
    # STEP 3: Prepare features data
    print(f"\nStep 3: Preparing feature data...")
    features_with_dates = features_df.copy()
    features_with_dates["Date"] = pd.to_datetime(features_with_dates["Date"], format='mixed')
    features_with_dates = features_with_dates.sort_values("Date").reset_index(drop=True)
    print(f"  ✓ Features prepared: {len(features_with_dates)} rows")
    
    # STEP 4: Generate training samples
    print(f"\nStep 4: Generating training samples...")
    X_samples = []
    Y_samples = []
    
    for edge_idx, (source_ticker, target_ticker, corr) in enumerate(edges):
        if edge_idx % max(1, len(edges)//10) == 0:
            print(f"  Processing edges: {edge_idx}/{len(edges)}")
        
        if source_ticker not in ticker_to_idx or target_ticker not in ticker_to_idx:
            continue
        
        source_idx = ticker_to_idx[source_ticker]
        target_idx = ticker_to_idx[target_ticker]
        
        # Get source and target data
        source_data = features_with_dates[
            features_with_dates["Symbol"] == source_ticker
        ].reset_index(drop=True)
        target_data = features_with_dates[
            features_with_dates["Symbol"] == target_ticker
        ].reset_index(drop=True)
        
        if source_data.empty or target_data.empty:
            continue
        
        # For each date (skip first since we need previous day)
        for i in range(1, len(source_data)):
            current_date = source_data.iloc[i]["Date"]
            
            # Get target data for current date
            target_current = target_data[target_data["Date"] == current_date]
            
            if target_current.empty:
                continue
            
            # Extract features
            source_prev_row = source_data.iloc[i - 1]
            target_curr_row = target_current.iloc[0]
            
            # Build X: source's previous day features + current day PCT-1
            source_features = {
                col: source_prev_row[col]
                for col in source_prev_row.index
                if col not in ["Date", "Symbol", "Unnamed: 0"]
            }
            
            source_features["current_PCT_1"] = source_data.iloc[i]["PCT-1"]
            source_features["source_node_idx"] = source_idx
            
            # Build Y: target's current day PCT-1 + target node index
            y_sample = {
                "target_PCT_1": target_curr_row["PCT-1"],
                "target_node_idx": target_idx
            }
            
            X_samples.append(source_features)
            Y_samples.append(y_sample)
    
    print(f"  ✓ Total samples generated: {len(X_samples)}")
    
    # STEP 5: Convert to DataFrames
    print(f"\nStep 5: Converting to DataFrames...")
    X_df = pd.DataFrame(X_samples)
    Y_df = pd.DataFrame(Y_samples)
    
    print(f"  ✓ X shape: {X_df.shape}")
    print(f"  ✓ Y shape: {Y_df.shape}")
    
    if X_df.empty or Y_df.empty:
        print("  ✗ No valid samples generated.")
        return
    
    # STEP 6: Random sampling
    print(f"\nStep 6: Random sampling...")
    if len(X_df) > n_samples:
        sample_indices = np.random.choice(len(X_df), n_samples, replace=False)
        X_df = X_df.iloc[sample_indices].reset_index(drop=True)
        Y_df = Y_df.iloc[sample_indices].reset_index(drop=True)
        print(f"  ✓ Sampled {n_samples} from {len(sample_indices)} available samples")
    
    # STEP 7: Display results
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nX DataFrame:")
    print(f"  Shape: {X_df.shape}")
    print(f"  Columns ({len(X_df.columns)}): {list(X_df.columns)[:15]}...")
    print(f"  Data types:\n{X_df.dtypes.value_counts()}")
    print(f"\n  First row (sample):")
    print(f"    source_node_idx: {X_df.iloc[0]['source_node_idx']}")
    print(f"    current_PCT_1: {X_df.iloc[0]['current_PCT_1']:.6f}")
    print(f"    Close: {X_df.iloc[0]['Close']:.2f}")
    
    print(f"\nY DataFrame:")
    print(f"  Shape: {Y_df.shape}")
    print(f"  Columns: {list(Y_df.columns)}")
    print(f"\n  Statistics:")
    print(Y_df.describe())
    
    print(f"\n✓ TEST COMPLETED SUCCESSFULLY!")
    print(f"\nDataset is ready for training:")
    print(f"  - X: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"  - Y: {Y_df.shape[0]} samples, 2 columns (target_PCT_1, target_node_idx)")

if __name__ == "__main__":
    test_get_dataset_standalone()
