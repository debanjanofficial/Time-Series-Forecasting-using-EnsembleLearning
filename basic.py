# Run this code and share the output
import pandas as pd

submission = pd.read_csv("submission.csv")
print(f"Submission file shape: {submission.shape}")
print("\nFirst 5 rows:")
print(submission.head())
print("\nLast 5 rows:")
print(submission.tail())
print("\nDescriptive statistics:")
print(submission.describe())
