# src/evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

import shap  # pip install shap

def load_data(fp: Path):
    df = pd.read_csv(fp)
    X = df.drop("fpd_15", axis=1)
    y = df["fpd_15"]
    return X, y

def load_model(fp: Path):
    return joblib.load(fp)

def evaluate(y_true, y_prob, threshold=0.5):
    # Binarize predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    roc_auc  = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    print(f"ROC-AUC:    {roc_auc:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1 Score:   {f1:.3f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (thr={threshold})")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.show()

def plot_feature_importance(model, X, use_shap=True):
    # 1) XGBoost built-in
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        names = X.columns
        idxs = np.argsort(importances)[::-1]

        plt.figure(figsize=(8,5))
        plt.barh(names[idxs], importances[idxs])
        plt.gca().invert_yaxis()
        plt.title("Feature Importances (XGBoost Gain)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    # 2) SHAP summary
    if use_shap:
        explainer = shap.Explainer(clf, X, feature_names=X.columns)
        shap_vals = explainer(X)
        shap.summary_plot(shap_vals, X, show=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=Path('C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/data/processed/loans_featured.csv'),
        help="Path to feature-engineered CSV"
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path('C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/models/xgb_model.pkl'),
        help="Path to your saved pipeline"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for confusion matrix"
    )
    args = p.parse_args()

    # Load
    X, y = load_data(args.data)
    pipeline = load_model(args.model)

    # Predict probabilities
    y_prob = pipeline.predict_proba(X)[:,1]

    # Evaluate
    evaluate(y, y_prob, threshold=args.threshold)
    plot_feature_importance(pipeline, X)
