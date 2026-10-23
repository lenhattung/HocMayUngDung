# Ô lệnh 10 -- Công bằng hơn: chọn ngưỡng tốt nhất cho TỪNG hàm mất mát rồi mới so sánh
from sklearn.model_selection import train_test_split

X_a, X_v, y_a, y_v = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                      random_state=SEED)       # tách 20% train làm tập kiểm định
qt_a = QuantileTransformer(output_distribution="normal", random_state=0).fit(X_a)
grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
rows = {}
for name in ["BCE", "BCE có trọng số", "FL gamma=2", "FL gamma=2, alpha=0.75"]:
    m = train_mlp(qt_a.transform(X_a), y_a, losses[name])
    p_v = predict_proba(m, qt_a.transform(X_v))
    t_best = grid[np.argmax([f1_score(y_v, (p_v >= t).astype(int)) for t in grid])]
    p_te = predict_proba(m, qt_a.transform(X_test))
    pred = (p_te >= t_best).astype(int)
    rows[name] = {"Ngưỡng t*": t_best, "F1 test": f1_score(y_test, pred),
                  "Recall": recall_score(y_test, pred), "PR-AUC": average_precision_score(y_test, p_te)}
print(pd.DataFrame(rows).T.round(3))
