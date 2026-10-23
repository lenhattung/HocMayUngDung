# Ô lệnh 12 -- Bảng tổng kết Chương 2-4 trên tập test (XGBoost)
from sklearn.metrics import recall_score, precision_score

final = {}
for sname, s in {"Không lấy mẫu": None, **samplers,
                 "Borderline-2": BorderlineSMOTE(kind="borderline-2", random_state=SEED)}.items():
    p = make(xgb_clf(), s).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    final[sname] = (p, 0.5)
    if s is None:
        final["Không lấy mẫu, t = 0,26"] = (p, 0.26)
summary = {}
for name, (p, t) in final.items():
    pred = (p >= t).astype(int)
    summary[name] = {"F1": f1_score(y_test, pred), "Recall": recall_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "PR-AUC": average_precision_score(y_test, p)}
print(pd.DataFrame(summary).T.round(3))
