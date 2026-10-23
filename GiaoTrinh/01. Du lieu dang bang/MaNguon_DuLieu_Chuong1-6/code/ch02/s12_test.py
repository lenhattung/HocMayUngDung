# Ô lệnh 12 -- Kiểm tra trên tập test (ngưỡng 0,5)
test_table = {}
for name, pipe in configs.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    test_table[name] = {"PR-AUC": average_precision_score(y_test, p),
                        "F1": f1_score(y_test, pred),
                        "Recall": recall_score(y_test, pred),
                        "Precision": precision_score(y_test, pred),
                        "Bị gắn cờ": pred.sum()}
test_df = pd.DataFrame(test_table).T
test_df["Bị gắn cờ"] = test_df["Bị gắn cờ"].astype(int)
print(test_df)
