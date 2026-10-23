# Ô lệnh 12 -- Tỷ lệ vỡ nợ theo trạng thái trả nợ tháng 9
rate = df.groupby("PAY_1")["default"].agg(["count", "mean"])
rate.columns = ["Số khách hàng", "Tỷ lệ vỡ nợ"]
print(rate)

fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.bar(rate.index.astype(str), rate["Tỷ lệ vỡ nợ"], color="#2a78d6")
ax.axhline(df["default"].mean(), color="gray", ls="--")   # đường tỷ lệ chung 22%
ax.set_xlabel("PAY_1")
ax.set_ylabel("Tỷ lệ vỡ nợ")
plt.savefig("figures/ch01_s12_rate.pdf", bbox_inches="tight")
