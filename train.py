from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


try:
  print("Loading Data...")
  train_df = pd.read_csv("train_preprocessed.csv")
  test_df = pd.read_csv("test_preprocessed.csv")
  print("Data loaded successfully.")
except Exception as e:
  print(e)

zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 'PSH Flag Count', 'ECE Flag Count']
all_low_importance_features = zero_importance_features
features_to_drop = ['Label', 'Binary_Label'] + zero_importance_features

# Separate X's and Y's
X_train = train_df.drop(features_to_drop, axis=1)
y_train = train_df['Binary_Label']
X_test = test_df.drop(features_to_drop, axis=1)
y_test = test_df['Binary_Label']


# DATA SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape} \n")
print(f"X_test_scaled shape: {X_test_scaled.shape} \n")

# Create the Random forest model
rain_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)

# Train the model    pip install seaborn
print("Training Random Forest...")
rain_forest_model.fit(X_train_scaled, y_train)
print("Training complete")
y_predictions = rain_forest_model.predict(X_test_scaled)

# Evaluate 
print("MODEL PERFORMANCE")
print(f"F1 Score: {f1_score(y_test, y_predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_predictions, target_names=['Attack', 'Benign']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predictions))


# ISOLATION FOREST
iso_forest = IsolationForest(
    contamination=0.17,  
    random_state=42,
    n_jobs=-1
)

print("Training Isolation Forest...")
iso_forest.fit(X_train_scaled)

y_pred_iso = iso_forest.predict(X_test_scaled)

y_pred_iso_binary = (y_pred_iso == 1).astype(int)

print("\n=== ISOLATION FOREST RESULTS ===")
print(f"F1 Score: {f1_score(y_test, y_pred_iso_binary):.4f}")
print(classification_report(y_test, y_pred_iso_binary, target_names=['Attack', 'Benign']))

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

# Store metrics for comparison
rf_f1 = f1_score(y_test, y_predictions)
iso_f1 = f1_score(y_test, y_pred_iso_binary)

rf_report = classification_report(y_test, y_predictions, output_dict=True)
iso_report = classification_report(y_test, y_pred_iso_binary, output_dict=True)

print(f"\n{'Metric':<25} {'Random Forest':<20} {'Isolation Forest':<20}")
print("-" * 65)
print(f"{'F1 Score':<25} {rf_f1:<20.4f} {iso_f1:<20.4f}")
print(f"{'Attack Precision':<25} {rf_report['0']['precision']:<20.4f} {iso_report['0']['precision']:<20.4f}")
print(f"{'Attack Recall':<25} {rf_report['0']['recall']:<20.4f} {iso_report['0']['recall']:<20.4f}")
print(f"{'Benign Precision':<25} {rf_report['1']['precision']:<20.4f} {iso_report['1']['precision']:<20.4f}")
print(f"{'Benign Recall':<25} {rf_report['1']['recall']:<20.4f} {iso_report['1']['recall']:<20.4f}")

# Save the model

with open("rf_model.pkl", 'wb') as f:
  pickle.dump(rain_forest_model, f)

with open('scaler.pkl', 'wb') as f:
  pickle.dump(scaler,f)

print("Model Saved")