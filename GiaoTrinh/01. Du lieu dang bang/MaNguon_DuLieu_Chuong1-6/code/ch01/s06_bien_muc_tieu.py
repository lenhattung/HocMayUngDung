# Ô lệnh 6 -- Biến mục tiêu: có bao nhiêu khách hàng vỡ nợ?
counts = df["default"].value_counts()
ratio = df["default"].value_counts(normalize=True)
print(pd.DataFrame({"Số lượng": counts, "Tỷ lệ": ratio}))

fig, ax = plt.subplots(figsize=(4, 2.6))
ax.bar(["0: không vỡ nợ", "1: vỡ nợ"], counts.sort_index().values,
       color=["#2a78d6", "#eb6834"], width=0.5)
ax.set_ylabel("Số khách hàng")
plt.savefig("figures/ch01_s06_target.pdf", bbox_inches="tight")  # Jupyter: plt.show()
