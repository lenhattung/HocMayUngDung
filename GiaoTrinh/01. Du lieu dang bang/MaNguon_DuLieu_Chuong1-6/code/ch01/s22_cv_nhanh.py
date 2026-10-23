# Ô lệnh 22 -- Viết gọn bằng cross_validate và tính nhiều chỉ số một lúc
from sklearn.model_selection import cross_validate

scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
           "F1": "f1", "Recall": "recall"}
res = cross_validate(logreg, X_train, y_train, cv=skf, scoring=scoring)

for name in scoring:
    s = res["test_" + name]                     # mảng 5 điểm số, mỗi fold một điểm
    print(f"{name:8s}: {s.mean():.4f} ± {s.std(ddof=1):.4f}")
