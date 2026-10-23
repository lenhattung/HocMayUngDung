# Ô lệnh 7 -- Kiểm chứng trên dữ liệu thật: trọng số w và ngưỡng 1/(1+w) cho cùng quyết định?
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
from scipy.stats import spearmanr

p1 = xgb_clf().fit(X_train, y_train).predict_proba(X_test)[:, 1]                       # w = 1
pw = xgb_clf(scale_pos_weight=ratio).fit(X_train, y_train).predict_proba(X_test)[:, 1]  # w = 3,52
t_eq = 1 / (1 + ratio)

def report(p, t):
    pred = (p >= t).astype(int)
    return {"F1": f1_score(y_test, pred), "Recall": recall_score(y_test, pred),
            "Precision": precision_score(y_test, pred), "PR-AUC": average_precision_score(y_test, p)}

print(pd.DataFrame({"w = 1, t = 0,5": report(p1, 0.5),
                    f"w = {ratio:.2f}, t = 0,5": report(pw, 0.5),
                    f"w = 1, t = {t_eq:.3f}": report(p1, t_eq)}).T.round(3))
print(f"\nTương quan hạng Spearman giữa hai bộ xác suất: {spearmanr(p1, pw).correlation:.4f}")
