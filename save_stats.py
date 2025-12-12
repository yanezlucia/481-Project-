import pandas as pd

train_df = pd.read_csv("train_preprocessed.csv")

zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                            'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 
                            'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 
                            'PSH Flag Count', 'ECE Flag Count']
features_to_drop = ['Label', 'Binary_Label'] + zero_importance_features

# Separate benign and attack samples
benign_samples = train_df[train_df['Binary_Label'] == 1].drop(features_to_drop, axis=1)
attack_samples = train_df[train_df['Binary_Label'] == 0].drop(features_to_drop, axis=1)

# Calculate statistics
benign_stats = benign_samples.describe()
attack_stats = attack_samples.describe()

# Save to JSON
benign_stats.to_json('benign_stats.json')
attack_stats.to_json('attack_stats.json')

print("Statistics saved!")
print(f"Benign samples: {len(benign_samples)}")
print(f"Attack samples: {len(attack_samples)}")