# src/train_classification.py
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_curve, roc_curve,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
import matplotlib.pyplot as plt

def load_splits(processed_dir: Path):
    train = pd.read_csv(processed_dir / "train.csv")
    val   = pd.read_csv(processed_dir / "val.csv")
    Xtr, ytr = train.drop(columns=["label"]).values, train["label"].values
    Xva, yva = val.drop(columns=["label"]).values,   val["label"].values
    feature_names = list(train.columns[:-1])
    return (Xtr, ytr, Xva, yva, feature_names)

def evaluate_binary(model, X, y, tag: str, outdir: Path):
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    roc_auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)  # PR-AUC

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ["Normal","Attack"])
    plt.yticks(ticks, ["Normal","Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(outdir / f"confusion_{tag}.png", dpi=140)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y, prob)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({tag})"); plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / f"roc_{tag}.png", dpi=140)
    plt.close(fig)

    # PR curve
    precs, recs, _ = precision_recall_curve(y, prob)
    fig = plt.figure()
    plt.plot(recs, precs, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve ({tag})"); plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / f"pr_{tag}.png", dpi=140)
    plt.close(fig)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc_auc, pr_auc=ap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed", help="Directory with train/val/test CSVs")
    ap.add_argument("--out_models", default="models", help="Where to save models")
    ap.add_argument("--out_report", default="report", help="Where to save plots/metrics")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_models = Path(args.out_models); out_models.mkdir(parents=True, exist_ok=True)
    out_report = Path(args.out_report); out_report.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, feat_names = load_splits(processed)


    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None, solver="lbfgs"),
        "rf":     RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced")
    }

    results = {}
    best_name, best_f1, best_model = None, -1.0, None
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        metrics = evaluate_binary(mdl, Xva, yva, tag=f"{name}_val", outdir=out_report)
        results[name] = metrics
        if metrics["f1"] > best_f1:
            best_name, best_f1, best_model = name, metrics["f1"], mdl

    # Save best
    joblib.dump({"model": best_model, "features": feat_names}, out_models / "clf_model.joblib")

    # Write metrics JSON
    with open(out_report / "classification_val_metrics.json", "w") as f:
        json.dump({"results": results, "best": best_name}, f, indent=2)

    print(f"✅ Trained. Best model on VAL: {best_name} (F1={best_f1:.3f})")
    print(f"Saved → {out_models / 'clf_model.joblib'}")
    print(f"Metrics/plots → {out_report}")

if __name__ == "__main__":
    main()

