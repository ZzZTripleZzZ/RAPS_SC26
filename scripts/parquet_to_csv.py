import pandas as pd
import sys

# Check if file path is provided
if len(sys.argv) < 3:
    print("Usage: python script_name.py input.parquet output.csv")
    sys.exit(1)

# Using pyarrow as the engine
df = pd.read_parquet(sys.argv[1], engine='pyarrow')

# Using fastparquet as the engine (if you've installed it)
# df = pd.read_parquet('path_to_file.parquet', engine='fastparquet')

print(df.head())

#df.to_csv(sys.argv[2], index=False)
df.head(1000).to_csv(sys.argv[2], index=False)
