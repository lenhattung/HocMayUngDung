# Ô lệnh 16 -- Một đối thủ đơn giản: giữ nguyên mô hình, chỉ dời ngưỡng
from sklearn.model_selection import cross_val_predict

# Dự đoán "out-of-fold": mỗi khách hàng train được dự đoán bởi mô hình CHƯA từng thấy họ
oof = cross_val_predict(xgb_clf(), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
f1s = [f1_score(y_train, (oof >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print(f"Ngưỡng cho F1 cao nhất (tính trên tập train): t* = {best_t:.2f}, F1 = {max(f1s):.3f}")

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(thresholds, f1s)
ax.axvline(best_t, ls="--", color="gray"); ax.axvline(0.5, ls=":", color="gray")
ax.set_xlabel("Ngưỡng t"); ax.set_ylabel("F1 (out-of-fold)")
plt.savefig("figures/ch02_s16_threshold.pdf", bbox_inches="tight")
