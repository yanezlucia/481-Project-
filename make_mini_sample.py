import pandas as pd

df_test = pd.read_csv("test_preprocessed.csv")

test_attacks = df_test[df_test["Binary_Label"] == 0]
test_benign = df_test[df_test["Binary_Label"] == 1]

print("=" * 60)
print("creating mini sample.")
n_samples = 50
sample_attacks = test_attacks.sample(n=n_samples)
sample_benign = test_benign.sample(n=n_samples)

# Join the two samples
sampled_df = pd.concat([sample_attacks, sample_benign])
sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)

sampled_df.to_csv("sampled_test_data.csv", index=False)
print("Finished creating mini sample.")