# Ô lệnh 17 -- So sánh trên tập test: SMOTE hay dời ngưỡng?
p_smote = configs["XGBoost + SMOTE"].predict_proba(X_test)[:, 1]
cases = {"XGBoost, t = 0,5": (proba_base, 0.5),
         "XGBoost + SMOTE, t = 0,5": (p_smote, 0.5),
         f"XGBoost, t = {best_t:.2f}".replace(".", ","): (proba_base, best_t)}
cmp = {}
for name, (p, t) in cases.items():
    pred = (p >= t).astype(int)
    cmp[name] = {"F1": f1_score(y_test, pred), "Recall": recall_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "PR-AUC": average_precision_score(y_test, p)}
print(pd.DataFrame(cmp).T.round(3))
