# Ô lệnh 25 -- Chênh lệch giữa hai mô hình có phải do may rủi? Kiểm định Wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from scipy.stats import wilcoxon

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)  # 15 lần chia
auc = {name: cross_val_score(models[name], X_train, y_train, cv=rcv, scoring="roc_auc")
       for name in ["LogReg", "XGBoost", "LightGBM"]}
for name, s in auc.items():
    print(f"{name:8s}: ROC-AUC = {s.mean():.4f} ± {s.std(ddof=1):.4f}")
print()
for a, b in [("XGBoost", "LogReg"), ("XGBoost", "LightGBM")]:
    d = auc[a] - auc[b]
    p = wilcoxon(auc[a], auc[b]).pvalue
    print(f"{a} - {b}: chênh lệch TB = {d.mean():+.4f}, thắng {np.sum(d > 0)}/15 lần,"
          f" p = {p:.1e}")
