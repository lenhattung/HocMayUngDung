# Ô lệnh 8 -- Vẽ xác suất của mô hình có trọng số theo mô hình không trọng số
grid = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.scatter(p1, pw, s=3, alpha=0.3, color="#2a78d6", label="6 000 khách hàng test")
ax.plot(grid, ratio * grid / (ratio * grid + 1 - grid), color="#eb6834", lw=2, label="công thức lý thuyết")
ax.plot([0, 1], [0, 1], ls=":", color="gray")
ax.set_xlabel("p (w = 1)"); ax.set_ylabel(f"p_w (w = {ratio:.2f})"); ax.legend(fontsize=7)
plt.savefig("figures/ch05_s08_pw.pdf", bbox_inches="tight")
