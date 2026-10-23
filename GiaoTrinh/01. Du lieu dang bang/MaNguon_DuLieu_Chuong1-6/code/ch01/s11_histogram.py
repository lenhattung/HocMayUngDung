# Ô lệnh 11 -- Vẽ phân bố (histogram) của bốn biến số
fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
for ax, col in zip(axes, num_cols):
    ax.hist(df[col], bins=40, color="#2a78d6")
    ax.set_title(col)
    ax.locator_params(axis="x", nbins=3)     # chỉ hiện khoảng 3 vạch số cho đỡ rối
plt.savefig("figures/ch01_s11_hist.pdf", bbox_inches="tight")
