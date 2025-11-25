import os
import pandas as pd



train_df = pd.read_csv("train_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")

# Separate X's and Y's
X_train = train_df.drop(['Label', 'Binary_Label'], axis=1)
y_train = train_df['Binary_Label']
X_test = test_df.drop(['Label', 'Binary_Label'], axis=1)
y_test = test_df['Binary_Label']

print(X_train.columns.tolist())

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
