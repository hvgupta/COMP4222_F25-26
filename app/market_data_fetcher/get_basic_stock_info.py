import kagglehub
import os
import shutil

# Download latest version (returns the directory path where files are extracted)
downloaded_dir = kagglehub.dataset_download("andrewmvd/sp-500-stocks")

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move the specific file
src_file = os.path.join(downloaded_dir, "sp500_companies.csv")
dst_file = os.path.join(current_dir, "sp500_companies.csv")

shutil.move(src_file, dst_file)

print(f"File moved to: {dst_file}")