from datasets import load_dataset
import pandas as pd 

dataset = load_dataset("HallowsYves/CPSC481-data")


# Testing to see if data was loaded
print(dataset["train"][:5])