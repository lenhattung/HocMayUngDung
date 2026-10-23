# Ô lệnh 9 -- Phân bố xác suất dự đoán trên tập test
probs = {"BCE": p_bce}
for name in ["FL gamma=2", "FL gamma=5"]:
    probs[name] = predict_proba(train_mlp(Xtr, y_train, losses[name]), Xte)

fig, ax = plt.subplots(figsize=(5, 2.9))
bins = np.linspace(0, 1, 41)
for (name, p_), c in zip(probs.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.hist(p_, bins=bins, histtype="step", lw=1.6, color=c, label=name)
ax.set_xlabel("Xác suất dự đoán (tập test)"); ax.set_ylabel("Số khách hàng"); ax.legend()
plt.savefig("figures/ch06_s09_proba.pdf", bbox_inches="tight")
print({k: round(float(v.mean()), 3) for k, v in probs.items()})
