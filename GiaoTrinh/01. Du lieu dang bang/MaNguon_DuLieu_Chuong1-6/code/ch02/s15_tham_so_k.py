# Ô lệnh 15 -- Ảnh hưởng của số láng giềng k (đánh giá chéo, mất khoảng 1 phút)
rows = []
for k in [1, 3, 5, 10, 20, 50]:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("smote", SMOTE(k_neighbors=k, random_state=SEED)),
                     ("model", xgb_clf())])
    r = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring)
    rows.append({"k": k, **{m: r["test_" + m].mean()
                            for m in ["PR-AUC", "F1", "Recall", "Precision"]}})
k_table = pd.DataFrame(rows).set_index("k")
print(k_table)

ax = k_table[["PR-AUC", "F1", "Recall", "Precision"]].plot(marker="o", figsize=(5, 2.8), logx=True)
ax.set_xticks(k_table.index); ax.set_xticklabels(k_table.index)
ax.set_ylabel("Điểm CV trung bình")
plt.savefig("figures/ch02_s15_k.pdf", bbox_inches="tight")
