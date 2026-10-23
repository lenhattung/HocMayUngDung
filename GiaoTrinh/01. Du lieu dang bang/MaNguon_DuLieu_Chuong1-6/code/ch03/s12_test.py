# Ô lệnh 12 -- Kiểm tra trên tập test (XGBoost, ngưỡng 0,5)
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score

test_rows, proba = {}, {}
for sname, s in strategies.items():
    pipe = make(xgb_clf(), s).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    proba[sname] = p
    pred = (p >= 0.5).astype(int)
    test_rows[sname] = {"PR-AUC": average_precision_score(y_test, p), "F1": f1_score(y_test, pred),
                        "Recall": recall_score(y_test, pred), "Precision": precision_score(y_test, pred)}
print(pd.DataFrame(test_rows).T)
