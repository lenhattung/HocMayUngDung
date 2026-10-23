# Ô lệnh 6 -- So sánh SMOTE, Borderline-1 và Borderline-2 (thư viện imblearn)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

samplers = {"SMOTE": SMOTE(random_state=SEED),
            "Borderline-1": BorderlineSMOTE(kind="borderline-1", random_state=SEED),
            "Borderline-2": BorderlineSMOTE(kind="borderline-2", random_state=SEED)}
synth = {}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True, sharey=True)
for ax, (name, s) in zip(axes, samplers.items()):
    Xr, yr = s.fit_resample(X_toy, y_toy)
    synth[name] = Xr[len(X_toy):]                     # các điểm mới nằm ở cuối mảng
    ax.scatter(*X_toy[y_toy == 0].T, s=6, color="#c9c8c2")
    ax.scatter(*X_min.T, s=12, color="#eb6834")
    ax.scatter(*synth[name].T, s=8, marker="x", color="#2a78d6", linewidth=0.7)
    ax.set_title(f"{name} ({len(synth[name])} điểm mới)")
plt.savefig("figures/ch03_s06_compare.pdf", bbox_inches="tight")
