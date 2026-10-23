# Ô lệnh 11 -- Thử thách: dữ liệu train có NHÃN SAI. Kỹ thuật nào chịu đựng tốt nhất?
from sklearn.metrics import average_precision_score, f1_score

rng = np.random.default_rng(SEED)
y_noisy = y_train.copy()
flip = rng.choice(np.where(y_train == 0)[0], size=int(0.05 * (y_train == 0).sum()), replace=False)
y_noisy.iloc[flip] = 1                     # 5% người KHÔNG vỡ nợ bị ghi nhầm thành vỡ nợ
print(f"Đã lật nhãn {len(flip)} khách hàng (chỉ trong tập train; tập test giữ nguyên)")

rows = {}
for sname, s in {"Không lấy mẫu": None, **samplers}.items():
    res = {}
    for tag, yy in [("sạch", y_train), ("nhiễu", y_noisy)]:
        p = make(xgb_clf(), s).fit(X_train, yy).predict_proba(X_test)[:, 1]
        res[f"PR-AUC {tag}"] = average_precision_score(y_test, p)
        res[f"F1 {tag}"] = f1_score(y_test, (p >= 0.5).astype(int))
    res["Giảm F1"] = res["F1 sạch"] - res["F1 nhiễu"]
    rows[sname] = res
print(pd.DataFrame(rows).T.round(3).to_string())
