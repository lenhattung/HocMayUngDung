# Ô lệnh 8 -- Dùng SMOTE có sẵn trong imbalanced-learn và vẽ kết quả
from imblearn.over_sampling import SMOTE

X_sm, y_sm = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X_toy, y_toy)
print("Sau SMOTE [lớp 0, lớp 1]:", np.bincount(y_sm))
syn = X_sm[len(X_toy):]                          # imblearn đặt các điểm mới ở cuối mảng

fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharex=True, sharey=True)
for ax, title in zip(axes, ["Dữ liệu gốc", "Sau SMOTE (k = 5)"]):
    ax.scatter(*X_toy[y_toy == 0].T, s=8, color="#c9c8c2", label="Lớp đa số")
    ax.scatter(*X_toy[y_toy == 1].T, s=14, color="#eb6834", label="Lớp thiểu số (gốc)")
    ax.set_title(title)
axes[1].scatter(*syn.T, s=10, marker="x", color="#2a78d6", label="Mẫu tổng hợp")
axes[1].legend(loc="lower right", fontsize=7)
plt.savefig("figures/ch02_s08_toy.pdf", bbox_inches="tight")
