import pandas as pd
import pickle

# Load your model and sample
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

sample_df = pd.read_csv('sampled_test_data.csv')

# Drop labels and zero-importance features
zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                            'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 
                            'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 
                            'PSH Flag Count', 'ECE Flag Count']
X = sample_df.drop(['Label', 'Binary_Label'] + zero_importance_features, axis=1)
y = sample_df['Binary_Label']

# Get predictions for all samples
X_scaled = scaler.transform(X)
predictions = model.predict_proba(X_scaled)
confidences = predictions.max(axis=1) * 100

# Show confidence distribution
print("Confidence distribution:")
print(f"Min confidence: {confidences.min():.2f}%")
print(f"Max confidence: {confidences.max():.2f}%")
print(f"Average confidence: {confidences.mean():.2f}%")
print(f"\nSamples with 100% confidence: {(confidences == 100).sum()} out of {len(confidences)}")
print(f"Samples below 90% confidence: {(confidences < 90).sum()}")