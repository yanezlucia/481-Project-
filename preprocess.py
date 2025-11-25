from datasets import load_dataset
import pandas as pd 


try:
    print("Loading Dataset...")
    
    dataset = load_dataset("HallowsYves/CPSC481-data")

    print("Converting to Dataframes... \n")
    df_train = dataset['test'].to_pandas()
    df_test = dataset['train'].to_pandas()

    # Create new Binary label columns for train and test
    df_train['Binary_Label'] = 0
    df_test['Binary_Label'] = 0

    df_train.loc[df_train['Label'] == 'Benign', 'Binary_Label'] = 1
    df_test.loc[df_test['Label'] == 'Benign', 'Binary_Label'] = 1

    df_train.to_csv('train_preprocessed.csv', index=False)
    df_test.to_csv('test_preprocessed.csv', index=False)

    print("Preprocessed data saved!")

except Exception as e:
    print(f"Something Went Wrong {e}")
