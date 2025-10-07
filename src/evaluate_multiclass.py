# src/evaluate_multiclass.py
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed_five")
    ap.add_argument("--model_path", default="models/clf_multiclass.joblib")
    ap.add_argument("--out_report", default="report")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_report = Path(args.out_report); out_report.mkdir(parents=True, exist_ok=True)

    # Load test split
    test = pd.read_csv(processed / "test.csv")
    Xte = test.drop(columns=["label"]).to_numpy()
    yte_str = test["label"].astype(str).to_numpy()

    # Load model bundle 
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    le = bundle.get("label_encoder", None)
    classes = bundle.get("classes", None)

    if le is None or classes is None:
        raise ValueError("Model bundle missing label_encoder/classes. Re-train with train_multiclass.py provided.")

    # Predict ints then convert to string
    ypred_int = model.predict(Xte)
    ypred_str = le.inverse_transform(ypred_int)
    class_names = classes  

    # Metrics
    acc = accuracy_score(yte_str, ypred_str)
    macro_f1 = f1_score(yte_str, ypred_str, average="macro", labels=class_names)
    per_class_arr = f1_score(yte_str, ypred_str, average=None, labels=class_names)
    per_class_f1 = {cls: float(f1) for cls, f1 in zip(class_names, per_class_arr)}

    # Confusion matrix
    cm = confusion_matrix(yte_str, ypred_str, labels=class_names)
    fig = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (test, 5-class)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_report / "confusion_test_5class.png", dpi=140)
    plt.close(fig)

    # Text report 
    report_txt = classification_report(yte_str, ypred_str, labels=class_names, digits=4, zero_division=0)
    with open(out_report / "classification_report_test_5class.txt", "w") as f:
        f.write(report_txt)

    metrics = {"accuracy": float(acc), "macro_f1": float(macro_f1), "per_class_f1": per_class_f1}
    with open(out_report / "multiclass_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Multiclass test metrics:", metrics)

if __name__ == "__main__":
    main()
