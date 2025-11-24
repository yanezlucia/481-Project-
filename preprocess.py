from datasets import load_dataset
import pandas as pd 

try:
    print("Loading Dataset...")
    
    dataset = load_dataset("HallowsYves/CPSC481-data")

    print("Converting to Dataframes...")
    df_train = dataset['test'].to_pandas()
    df_test = dataset['train'].to_pandas()

    print("Column names: ")
    print(df_train.columns.tolist())
    print(f"\n Total Columns: {len(df_train.columns )}")
except Exception as e:
    print(f"Something Went Wrong {e}")
