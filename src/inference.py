# src/inference.py
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

# ---------- feature engineering (must match data_prep.py) ----------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bytes_total"] = d["src_bytes"].clip(lower=0) + d["dst_bytes"].clip(lower=0)
    d["bytes_ratio_src"] = d["src_bytes"].clip(lower=0) / (d["bytes_total"] + 1)
    d["log_bytes_total"] = np.log1p(d["bytes_total"])
    d["count_per_srv"] = d["count"] / (d["srv_count"] + 1)
    d["duration_per_conn"] = d["duration"] / (d["count"] + 1)
    web = {"http","http_443","http_8001","www"}
    mail = {"smtp","imap4","pop_2","pop_3"}
    ftp  = {"ftp","ftp_data"}
    dns  = {"domain","domain_u"}
    d["service_group"] = np.where(d["service"].isin(web), "web",
                          np.where(d["service"].isin(mail), "mail",
                          np.where(d["service"].isin(ftp),  "ftp",
                          np.where(d["service"].isin(dns),  "dns", "other"))))
    return d

# ---------- anomaly score helper (higher = more anomalous) ----------
def anomaly_scores(model, X):
    if hasattr(model, "decision_function"):
        # For IF/OCSVM: higher (normal) -> lower anomaly; invert
        return -model.decision_function(X)
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    if hasattr(model, "predict"):
        # Fallback: -1 (outlier) -> 1.0, 1 (inlier) -> 0.0
        y = model.predict(X)
        return (y == -1).astype(float)
    raise ValueError("Model does not expose a scoring API for anomaly detection.")

def load_bundle(preproc_path: Path):
    bundle = joblib.load(preproc_path)
    pre = bundle["preprocess"]
    meta = bundle["meta"]
    feature_cols = meta["feature_cols"]
    return pre, feature_cols, meta

def prepare_input(df_in: pd.DataFrame, feature_cols, pre):
    # ensure column names normalized like in prep
    df = df_in.copy()
    df.columns = df.columns.str.strip().str.lower()

    # engineer features
    df = engineer_features(df)

    # we only need the columns the preprocessor was trained on
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df[feature_cols].copy()
    X_trans = pre.transform(X)
    return df, X_trans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV (raw-like schema: duration, protocol_type, service, flag, src_bytes, dst_bytes, count, srv_count, serror_rate)")
    ap.add_argument("--task", choices=["classification","anomaly"], default="classification")
    ap.add_argument("--preproc", default="models/scalers_encoders.joblib", help="Path to fitted preprocessor artifact")
    ap.add_argument("--model", required=True, help="Path to model: e.g., models/clf_model.joblib OR models/anomaly_iforest.joblib")
    ap.add_argument("--out_csv", default=None, help="Optional: where to write predictions CSV")
    args = ap.parse_args()

    input_path = Path(args.input)
    preproc_path = Path(args.preproc)
    model_path = Path(args.model)

    # read input
    df_in = pd.read_csv(input_path)

    # load preprocessor + meta (includes feature_cols)
    pre, feature_cols, meta = load_bundle(preproc_path)

    # transform input using same pipeline
    df_raw_like, X = prepare_input(df_in, feature_cols, pre)

    # load model (could be plain sklearn estimator OR a bundle with label_encoder/classes for multiclass)
    mdl_bundle = joblib.load(model_path)

    # support both plain estimator and bundle dict
    if isinstance(mdl_bundle, dict) and "model" in mdl_bundle:
        model = mdl_bundle["model"]
        label_encoder = mdl_bundle.get("label_encoder", None)  # present for multiclass
        classes = mdl_bundle.get("classes", None)
    else:
        model = mdl_bundle
        label_encoder = None
        classes = None

    if args.task == "classification":
        out = pd.DataFrame(index=df_in.index)
        # predict labels
        y_pred = model.predict(X)
        if label_encoder is not None:
            # multiclass bundle: map ints -> strings
            y_label = label_encoder.inverse_transform(y_pred)
            out["pred_label"] = y_label
            # proba per class (if available)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                for i, cls in enumerate(classes):
                    out[f"proba_{cls}"] = proba[:, i]
        else:
            # binary: map 0/1 to strings if you prefer
            out["pred_label"] = y_pred
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                # positive class probability column name
                pos_index = 1 if proba.shape[1] == 2 else np.argmax(proba.mean(axis=0))
                out["pred_proba"] = proba[:, pos_index]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                out["pred_proba"] = 1 / (1 + np.exp(-scores))  # sigmoid fallback
        result = out

    else:  # anomaly
        scores = anomaly_scores(model, X)
        result = pd.DataFrame({"anomaly_score": scores}, index=df_in.index)

    # optional: join a few input columns for readability
    cols_show = [c for c in ["protocol_type","service","flag","src_bytes","dst_bytes","duration","count","srv_count"] if c in df_raw_like.columns]
    result = pd.concat([df_raw_like[cols_show].reset_index(drop=True), result.reset_index(drop=True)], axis=1)

    # output
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out_csv, index=False)
    else:
        # pretty print small head
        print(result.head(10).to_string(index=False))

if _name_ == "_main_":
    main()
