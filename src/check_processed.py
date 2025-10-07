import pandas as pd
import joblib
import json

# 1. Load processed splits
train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
test  = pd.read_csv("data/processed/test.csv")

print("Shapes:")
print(" Train:", train.shape)
print(" Val  :", val.shape)
print(" Test :", test.shape)
print()

# 2. Check class balance (after SMOTE)
print("Train label distribution (after SMOTE):")
print(train['label'].value_counts(normalize=True))
print()

print("Val label distribution:")
print(val['label'].value_counts(normalize=True))
print()

print("Test label distribution:")
print(test['label'].value_counts(normalize=True))
print()

# 3. Check feature names match artifact
artifact = joblib.load("models/scalers_encoders.joblib")
feature_names = artifact["meta"]["feature_names_out"]

print("First 10 feature columns:", feature_names[:10])
print("... total features:", len(feature_names))

# 4. Compare to CSV columns
print("CSV columns match artifact?", list(train.columns[:-1]) == list(feature_names))

# 5. Show prep stats JSON
with open("data/processed/prep_stats.json") as f:
    stats = json.load(f)

print("\nPrep stats summary:")
print(stats)
