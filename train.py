from sklearn.preprocessing import StandardScaler
import pandas as pd


# Load Data
train_df = pd.read_csv("train_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")


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


"""
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
"""