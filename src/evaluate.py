# src/evaluate.py
import argparse, json
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
                             confusion_matrix)
import matplotlib.pyplot as plt

def eval_test(model, X, y, outdir: Path):
    prob = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob)
    }
    # Plots
    cm = confusion_matrix(y, pred)
    fig = plt.figure(); plt.imshow(cm); plt.title("Confusion (test)"); plt.colorbar()
    plt.xticks([0,1], ["Normal","Attack"]); plt.yticks([0,1], ["Normal","Attack"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(outdir / "confusion_test.png", dpi=140); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, prob)
    fig = plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.title("ROC (test)")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
    fig.savefig(outdir / "roc_test.png", dpi=140); plt.close(fig)

    precs, recs, _ = precision_recall_curve(y, prob)
    fig = plt.figure(); plt.plot(recs,precs); plt.title("PR (test)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout()
    fig.savefig(outdir / "pr_test.png", dpi=140); plt.close(fig)

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model_path", default="models/clf_model.joblib")
    ap.add_argument("--out_report", default="report")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_report = Path(args.out_report); out_report.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(processed / "test.csv")
    Xte, yte = test.drop(columns=["label"]).values, test["label"].values

    bundle = joblib.load(args.model_path)
    model = bundle["model"]

    metrics = eval_test(model, Xte, yte, outdir=out_report)
    with open(out_report / "classification_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Test metrics:", metrics)

if __name__ == "__main__":
    main()
    
