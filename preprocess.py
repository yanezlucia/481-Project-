from datasets import load_dataset
import pandas as pd 

dataset = load_dataset("HallowsYves/CPSC481-data")

# Merge Data into two dataframes
df_train = pd.concat([dataset["train"].to_pandas()])
df_testing = pd.concat([dataset["test"].to_pandas()])
