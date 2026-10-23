# Ô lệnh 24 -- Đánh giá MỘT LẦN trên tập test (sau khi đã so sánh xong bằng CV)
rows = {}
for name, m in models.items():
    m.fit(X_train, y_train)                     # học lại trên toàn bộ tập train
    p = m.predict_proba(X_test)[:, 1]
    rows[name] = {"ROC-AUC": roc_auc_score(y_test, p),
                  "PR-AUC": average_precision_score(y_test, p),
                  "F1": f1_score(y_test, (p >= 0.5).astype(int)),
                  "Recall": recall_score(y_test, (p >= 0.5).astype(int))}
print(pd.DataFrame(rows).T)
