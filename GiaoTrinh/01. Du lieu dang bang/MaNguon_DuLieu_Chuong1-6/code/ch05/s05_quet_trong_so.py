# Ô lệnh 5 -- Quét trọng số lớp dương w cho XGBoost (đánh giá chéo, ~40 giây)
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "F1": "f1",
           "Recall": "recall", "Precision": "precision"}
rows = []
for w_ in [1, 1.5, 2, 3, round(ratio, 2), 5, 8]:
    r = cross_validate(xgb_clf(scale_pos_weight=w_), X_train, y_train, cv=skf, scoring=scoring)
    rows.append({"w": w_, **{k: r["test_" + k].mean() for k in scoring}})
sweep = pd.DataFrame(rows).set_index("w")
print(sweep)

ax = sweep[["PR-AUC", "F1", "Recall", "Precision"]].plot(marker="o", figsize=(5.2, 2.9))
ax.axvline(ratio, ls="--", color="gray")
ax.set_xlabel("Trọng số lớp dương w (scale_pos_weight)"); ax.set_ylabel("Điểm CV trung bình")
plt.savefig("figures/ch05_s05_sweep.pdf", bbox_inches="tight")
