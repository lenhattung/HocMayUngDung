# Ô lệnh 6 -- Đánh giá MLP (hàm mất mát BCE) trên tập test
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, brier_score_loss)

def report(p, y_true=y_test, t=0.5):
    pred = (p >= t).astype(int)
    return {"ROC-AUC": roc_auc_score(y_true, p), "PR-AUC": average_precision_score(y_true, p),
            "F1": f1_score(y_true, pred), "Recall": recall_score(y_true, pred),
            "Precision": precision_score(y_true, pred), "Brier": brier_score_loss(y_true, p),
            "p TB": p.mean()}

p_bce = predict_proba(mlp_bce, Xte)
print(pd.Series(report(p_bce)).round(3).to_string())
