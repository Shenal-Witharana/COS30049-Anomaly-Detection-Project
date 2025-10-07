# src/data_prep.py
# Prepares train/val/test and a fitted preprocessing artifact for simplified CICIDS.
# Usage examples:
#   python -m src.data_prep --raw data/raw/cicids_simplified --out data/processed --target binary --smote
#   python -m src.data_prep --raw data/raw/cicids_simplified --out data/processed --target fiveclass

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


def load_simplified_cicids(raw_dir: Path) -> pd.DataFrame:
    """Load the simplified dataset + map labels → (binary, 5-class)."""
    df = pd.read_csv(raw_dir / "basic_data_4.csv")
    df.columns = df.columns.str.strip().str.lower()

    expected = {
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "label"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # raw attack label -> dos, probe, r2l, u2r
    labmap = pd.read_csv(raw_dir / "label_category_map.csv")
    labmap.columns = labmap.columns.str.strip().str.lower()
    raw_col = "raw_label" if "raw_label" in labmap.columns else labmap.columns[0]
    cat_col = "category"  if "category"  in labmap.columns else labmap.columns[1]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    labmap[raw_col] = labmap[raw_col].astype(str).str.strip().str.lower()

    df = df.merge(labmap[[raw_col, cat_col]], left_on="label", right_on=raw_col, how="left")

    
    df[cat_col] = np.where(
        (df["label"] == "normal") & (df[cat_col].isna()),
        "normal",
        df[cat_col]
    )

    # Targets
    df["label_binary"] = (df[cat_col].str.lower() != "normal").astype(int)
    df["label_5class"] = df[cat_col].str.lower()

    return df


def split_stratified(df: pd.DataFrame, label_col: str, seed: int = 42):
    """70/15/15 stratified split."""
    train_df, temp_df = train_test_split(
        df, train_size=0.70, stratify=df[label_col], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[label_col], random_state=seed
    )
    return train_df, val_df, test_df


def build_preprocessor(X_train: pd.DataFrame):
    """Impute+Scale numeric, Impute+OHE categoricals. Return transformer and column lists."""
    num_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate"]
    cat_cols = ["protocol_type", "service", "flag"]

    
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols


def get_feature_names(pre: ColumnTransformer, num_cols, cat_cols):
    """Recover final feature names after OHE."""
    num_names = list(num_cols)
    cat_encoder = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist() if len(cat_cols) else []
    return num_names + cat_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Folder with basic_data_4.csv and label_category_map.csv")
    p.add_argument("--out", required=True, help="Output folder for processed CSVs")
    p.add_argument("--target", choices=["binary", "fiveclass"], default="binary",
                   help="Which label to stratify on and save as 'label'")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE to TRAIN only (binary task recommended)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + label mapping
    df = load_simplified_cicids(raw_dir)

    # 2) Choose target
    label_col = "label_binary" if args.target == "binary" else "label_5class"
    if label_col not in df.columns:
        raise ValueError(f"Target column '{label_col}' not found.")

    # 3) Split (stratified)
    train_df, val_df, test_df = split_stratified(df, label_col=label_col, seed=args.seed)

    # 4) Separate X/y
    feature_cols = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()

    X_val   = val_df[feature_cols].copy()
    y_val   = val_df[label_col].copy()

    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[label_col].copy()

    # 5) Preprocess (fit on TRAIN only)
    pre, num_cols, cat_cols = build_preprocessor(X_train)
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)

    # 6) SMOTE (TRAIN only). 
    if args.smote:
        if label_col != "label_binary":
            raise ValueError("--smote is intended for binary target. Use --target binary.")
        if SMOTE is None:
            raise ImportError("imblearn not installed; cannot use --smote")
        sm = SMOTE(random_state=args.seed)
        Xtr, y_train = sm.fit_resample(Xtr, y_train)

    # 7) Save processed CSVs (feature names + label)
    feat_names = get_feature_names(pre, num_cols, cat_cols)
    train_out = pd.DataFrame(Xtr, columns=feat_names)
    val_out   = pd.DataFrame(Xva, columns=feat_names)
    test_out  = pd.DataFrame(Xte, columns=feat_names)

    train_out["label"] = y_train.values
    val_out["label"]   = y_val.values
    test_out["label"]  = y_test.values

    train_out.to_csv(out_dir / "train.csv", index=False)
    val_out.to_csv(out_dir / "val.csv", index=False)
    test_out.to_csv(out_dir / "test.csv", index=False)

    # 8) Save preprocessing artifact + meta
    meta = {
        "target_mode": args.target,
        "feature_cols": feature_cols,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "feature_names_out": feat_names,
        "split": {"train": len(train_out), "val": len(val_out), "test": len(test_out)},
        "class_balance": {
            "train": pd.Series(y_train).value_counts(normalize=True).to_dict(),
            "val":   pd.Series(y_val).value_counts(normalize=True).to_dict(),
            "test":  pd.Series(y_test).value_counts(normalize=True).to_dict(),
        },
    }
    joblib.dump({"preprocess": pre, "meta": meta}, models_dir / "scalers_encoders.joblib")

    # 9) JSON for quick glance
    with open(out_dir / "prep_stats.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Data prep complete")
    print(f"Processed CSVs → {out_dir}")
    print("Transformer artifact → models/scalers_encoders.joblib")


if __name__ == "__main__":
    main()
