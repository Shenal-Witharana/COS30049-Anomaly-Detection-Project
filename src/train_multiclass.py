import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

def load_splits(processed_dir: Path):
    train = pd.read_csv(processed_dir / "train.csv")
    val   = pd.read_csv(processed_dir / "val.csv")
    Xtr = train.drop(columns=["label"]).to_numpy()
    ytr = train["label"].astype(str).to_numpy()
    Xva = val.drop(columns=["label"]).to_numpy()
    yva = val["label"].astype(str).to_numpy()
    feature_names = list(train.columns[:-1])
    return Xtr, ytr, Xva, yva, feature_names

def eval_multiclass(model, X, y_true_str, le, outdir: Path, tag: str):
    y_pred_int = model.predict(X)
    y_pred_str = le.inverse_transform(y_pred_int)

    class_names = le.classes_.tolist()  # consistent class order

    acc = accuracy_score(y_true_str, y_pred_str)
    macro_f1 = f1_score(y_true_str, y_pred_str, average="macro", labels=class_names)

    # Per-class F1 (ordered by class_names)
    per_class_f1_arr = f1_score(y_true_str, y_pred_str, average=None, labels=class_names)
    per_class = {cls: float(f1) for cls, f1 in zip(class_names, per_class_f1_arr)}

    # Confusion matrix
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(outdir / f"confusion_{tag}.png", dpi=140)
    plt.close(fig)

    # Text report
    report_txt = classification_report(y_true_str, y_pred_str, labels=class_names, digits=4)
    with open(outdir / f"classification_report_{tag}.txt", "w") as f:
        f.write(report_txt)

    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "per_class_f1": per_class}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed_five")
    ap.add_argument("--out_models", default="models")
    ap.add_argument("--out_report", default="report")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_models = Path(args.out_models); out_models.mkdir(parents=True, exist_ok=True)
    out_report = Path(args.out_report); out_report.mkdir(parents=True, exist_ok=True)

    Xtr, ytr_str, Xva, yva_str, feat_names = load_splits(processed)

    # Encode labels 
    le = LabelEncoder()
    ytr = le.fit_transform(ytr_str)
    yva = le.transform(yva_str)
    class_names = le.classes_.tolist()  
    models = {
        "logreg_multi": LogisticRegression(max_iter=4000),
        "rf_multi":     RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1)
    }

    results = {}
    best_name, best_score, best_model = None, -1.0, None
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        metrics = eval_multiclass(
            model=mdl,
            X=Xva,
            y_true_str=yva_str,                 
            le=le,
            outdir=out_report,
            tag=f"{name}_val"
        )
        results[name] = metrics
        if metrics["macro_f1"] > best_score:
            best_name, best_score, best_model = name, metrics["macro_f1"], mdl

    # Save best model
    joblib.dump(
        {"model": best_model, "features": feat_names, "label_encoder": le, "classes": class_names},
        out_models / "clf_multiclass.joblib"
    )

    with open(out_report / "multiclass_val_metrics.json", "w") as f:
        json.dump({"results": results, "best": best_name}, f, indent=2)

    print(f"✅ Multiclass trained. Best on VAL: {best_name} (macro-F1={best_score:.3f})")
    print(f"Saved → {out_models / 'clf_multiclass.joblib'}")
    print(f"Metrics/plots → {out_report}")

if __name__ == "__main__":
    main()
