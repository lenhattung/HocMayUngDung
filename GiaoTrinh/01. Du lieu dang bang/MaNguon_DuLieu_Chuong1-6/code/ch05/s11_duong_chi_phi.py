# Ô lệnh 11 -- Vẽ tổng chi phí theo ngưỡng (C_FN/C_FP = 5)
ts = np.linspace(0.02, 0.9, 89)
costs = [total_cost(p1, t, 5)[2] for t in ts]
print(f"Ngưỡng có chi phí thấp nhất trên tập test: {ts[np.argmin(costs)]:.2f}"
      f" (lý thuyết: {1 / 6:.3f})")

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(ts, costs)
ax.axvline(1 / 6, ls="--", color="gray"); ax.axvline(0.5, ls=":", color="gray")
ax.set_xlabel("Ngưỡng t"); ax.set_ylabel("Tổng chi phí (C_FP = 1)")
plt.savefig("figures/ch05_s11_cost.pdf", bbox_inches="tight")
