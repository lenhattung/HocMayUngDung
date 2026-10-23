# Ô lệnh 8 -- Trên Credit Default: phân bố r_i và ngân sách ADASYN đi về đâu
Z_train = StandardScaler().fit_transform(X_train)
rc, _, gc, Gc = adasyn_weights(Z_train, y_train.values)
tab = pd.DataFrame({"% khách hàng vỡ nợ": pd.Series(rc).value_counts(normalize=True).sort_index() * 100,
                    "% điểm tổng hợp": pd.Series(gc).groupby(rc).sum() / gc.sum() * 100})
tab.index.name = "r_i"
print(tab.round(1))

fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
ax[0].bar(tab.index, tab.iloc[:, 0], width=0.15, color="#2a78d6")
ax[0].set_title("Khách hàng vỡ nợ theo r_i"); ax[0].set_ylabel("%")
ax[1].bar(tab.index, tab.iloc[:, 1], width=0.15, color="#eb6834")
ax[1].set_title("Điểm tổng hợp sinh ra từ đâu")
for a in ax:
    a.set_xlabel("r_i")
plt.savefig("figures/ch04_s08_credit_r.pdf", bbox_inches="tight")
