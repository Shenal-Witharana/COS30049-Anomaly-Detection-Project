# src/train_anomaly.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def precision_recall_at_k(y_true, scores, k=0.05):
    n = len(scores)
    topk = max(1, int(n * k))
    idx = np.argsort(scores)[::-1]  
    top_idx = idx[:topk]
    y_top = y_true[top_idx]
    prec_at_k = y_top.mean() if topk > 0 else 0.0
    recall_at_k = y_top.sum() / max(1, y_true.sum())
    return float(prec_at_k), float(recall_at_k)

def fit_isoforest(X):
    return IsolationForest(n_estimators=300, max_samples="auto", random_state=42, n_jobs=-1).fit(X)

def score_model(model, X):
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    elif hasattr(model, "decision_function"):
        return -model.decision_function(X)
    else:
        raise ValueError("Model has no scoring method.")

def evaluate_scores(y, scores):
    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    p5, r5 = precision_recall_at_k(y, scores, k=0.05)
    p1, r1 = precision_recall_at_k(y, scores, k=0.01)
    return {"roc_auc": float(roc), "pr_auc": float(ap),
            "precision@5%": p5, "recall@5%": r5,
            "precision@1%": p1, "recall@1%": r1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out_models", default="models")
    ap.add_argument("--out_report", default="report")
    ap.add_argument("--algo", choices=["iforest","ocsvm","both"], default="iforest")
    ap.add_argument("--save_scores", action="store_true", help="Save anomaly scores for val/test to CSV for plotting/app")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_models = Path(args.out_models); out_models.mkdir(parents=True, exist_ok=True)
    out_report = Path(args.out_report); out_report.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed / "train.csv")
    val   = pd.read_csv(processed / "val.csv")
    test  = pd.read_csv(processed / "test.csv")

    Xtr = train.loc[train["label"] == 0].drop(columns=["label"]).to_numpy()

    Xva = val.drop(columns=["label"]).to_numpy()
    yva = val["label"].to_numpy(dtype=int)

    Xte = test.drop(columns=["label"]).to_numpy()
    yte = test["label"].to_numpy(dtype=int)

    bundles = {}
    metrics = {}

    if args.algo in ("iforest","both"):
        if_model = fit_isoforest(Xtr)
        s_val = score_model(if_model, Xva)
        s_tst = score_model(if_model, Xte)
        metrics["iforest_val"] = evaluate_scores(yva, s_val)
        metrics["iforest_test"] = evaluate_scores(yte, s_tst)
        if args.save_scores:
            pd.DataFrame({"score": s_val, "label": yva}).to_csv(out_report / "anomaly_scores_val_iforest.csv", index=False)
            pd.DataFrame({"score": s_tst, "label": yte}).to_csv(out_report / "anomaly_scores_test_iforest.csv", index=False)
        joblib.dump({"model": if_model}, out_models / "anomaly_iforest.joblib")
        bundles["iforest"] = "models/anomaly_iforest.joblib"

    if args.algo in ("ocsvm","both"):
        oc_model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Xtr)
        s_val = score_model(oc_model, Xva)
        s_tst = score_model(oc_model, Xte)
        metrics["ocsvm_val"] = evaluate_scores(yva, s_val)
        metrics["ocsvm_test"] = evaluate_scores(yte, s_tst)
        if args.save_scores:   
            pd.DataFrame({"score": s_val, "label": yva}).to_csv(out_report / "anomaly_scores_val_ocsvm.csv", index=False)
            pd.DataFrame({"score": s_tst, "label": yte}).to_csv(out_report / "anomaly_scores_test_ocsvm.csv", index=False)
        joblib.dump({"model": oc_model}, out_models / "anomaly_ocsvm.joblib")
        bundles["ocsvm"] = "models/anomaly_ocsvm.joblib"

    with open(out_report / "anomaly_metrics.json", "w") as f:
        json.dump({"bundles": bundles, "metrics": metrics}, f, indent=2)

    print("✅ Anomaly training complete.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

