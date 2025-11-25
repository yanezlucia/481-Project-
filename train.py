from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


try:
  print("Loading Data...")
  train_df = pd.read_csv("train_preprocessed.csv")
  test_df = pd.read_csv("test_preprocessed.csv")
  print("Data loaded successfully.")
except Exception as e:
  print(e)

# Separate X's and Y's
X_train = train_df.drop(['Label', 'Binary_Label'], axis=1)
y_train = train_df['Binary_Label']
X_test = test_df.drop(['Label', 'Binary_Label'], axis=1)
y_test = test_df['Binary_Label']


# DATA SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape} \n")
print(f"X_test_scaled shape: {X_test_scaled.shape} \n")

# Create the Random forest model
rain_forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)

# Train the model
print("Training Random Forest...")
rain_forest_model.fit(X_train, y_train)
print("Training complete")


y_predictions = rain_forest_model.predict(X_test_scaled)

# Evaluate 
print("MODEL PERFORMANCE")
print(f"F1 Score: {f1_score(y_test, y_predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_predictions, target_names=['Attack', 'Benign']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predictions))