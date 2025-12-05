from datasets import load_dataset
import pandas as pd 


try:
    print("Loading Dataset...")
    
    dataset = load_dataset("HallowsYves/CPSC481-data")

    print("Converting to Dataframes... \n")
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    # Create new Binary label columns for train and test
    df_train['Binary_Label'] = 0
    df_test['Binary_Label'] = 0

    df_train.loc[df_train['Label'] == 'Benign', 'Binary_Label'] = 1
    df_test.loc[df_test['Label'] == 'Benign', 'Binary_Label'] = 1

    # Mini sample for testing
    print("Getting mini sample from dataset...")
    test_attacks = df_test[df_test['Binary_Label'] == 0]
    test_benign = df_test[df_test['Binary_Label'] == 1]

    n_samples = 50
    sample_attacks = test_attacks.sample(n=min(n_samples, len(test_attacks)), random_state=42)
    sample_benign = test_benign.sample(n=min(n_samples, len(test_benign)), random_state=42)

    sampled_df = pd.concat([sample_attacks, sample_benign])
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    sampled_df.to_csv('sampled_test_data.csv', index=False)
    print(f"Saved mini sample from dataset. Can be found at sampled_test_data.csv")

    df_train.to_csv('train_preprocessed.csv', index=False)
    df_test.to_csv('test_preprocessed.csv', index=False)

    print("Preprocessed data saved!")

except Exception as e:
    print(f"Something Went Wrong {e}")
