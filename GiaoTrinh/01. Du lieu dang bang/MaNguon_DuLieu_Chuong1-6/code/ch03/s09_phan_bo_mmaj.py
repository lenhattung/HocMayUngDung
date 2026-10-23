# Ô lệnh 9 -- Phân bố số láng giềng thuộc lớp đa số (m') của mọi khách hàng vỡ nợ
counts = pd.Series(mmaj_c).value_counts().sort_index()
print(counts.to_dict())

colors = ["#1baf7a"] * 5 + ["#eb6834"] * 5 + ["#e34948"]   # SAFE: 0-4, DANGER: 5-9, NOISE: 10
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.bar(counts.index, counts.values, color=[colors[i] for i in counts.index])
ax.set_xticks(range(11))
ax.set_xlabel("m' = số láng giềng (trong 10) thuộc lớp KHÔNG vỡ nợ")
ax.set_ylabel("Số khách hàng vỡ nợ")
plt.savefig("figures/ch03_s09_mmaj.pdf", bbox_inches="tight")
