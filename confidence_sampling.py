import pandas as pd 
import pickle
import numpy as np

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df_test = pd.read_csv('test_preprocessed.csv')

zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                            'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 
                            'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 
                            'PSH Flag Count', 'ECE Flag Count']
X = df_test.drop(['Label', 'Binary_Label'] + zero_importance_features, axis=1)

print("\n" +"=" * 60)
print("Calculating confidence for all test samples...")
print("=" * 60)

X_scaled = scaler.transform(X)
predictions = model.predict_proba(X_scaled)
confidences = predictions.max(axis=1) * 100

df_test['confidence'] = confidences

print(f"\nConfidence distribution across full test set:")
print(f"Min: {confidences.min():.2f}%")
print(f"Max: {confidences.max():.2f}%")
print(f"Average: {confidences.mean():.2f}%")
print(f"Median: {np.median(confidences):.2f}%")

benign = df_test[df_test['Binary_Label'] == 1]
attacks = df_test[df_test['Binary_Label'] == 0]

def sample_by_confidence_range(df, n_total):
    """Sample across different confidence ranges"""
    samples = []
    
    low_conf = df[df['confidence'] < 80]
    if len(low_conf) > 0:
        n = min(int(n_total * 0.3), len(low_conf))
        samples.append(low_conf.sample(n=n))
    
    med_conf = df[(df['confidence'] >= 80) & (df['confidence'] < 95)]
    if len(med_conf) > 0:
        n = min(int(n_total * 0.4), len(med_conf))
        samples.append(med_conf.sample(n=n))
    
    high_conf = df[df['confidence'] >= 95]
    if len(high_conf) > 0:
        n = min(int(n_total * 0.3), len(high_conf))
        samples.append(high_conf.sample(n=n))
    
    return pd.concat(samples)

# Sample both classes
n_per_class = 250
benign_sample = sample_by_confidence_range(benign, n_per_class)
attack_sample = sample_by_confidence_range(attacks, n_per_class)

# Combine and shuffle
sampled_df = pd.concat([benign_sample, attack_sample])
sampled_df = sampled_df.drop('confidence', axis=1) 
sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)

# Save
sampled_df.to_csv('sampled_test_data.csv', index=False)

print(f"\nCreated diverse sample with {len(sampled_df)} total samples")
print(f"- Benign: {len(benign_sample)}")
print(f"- Attack: {len(attack_sample)}")