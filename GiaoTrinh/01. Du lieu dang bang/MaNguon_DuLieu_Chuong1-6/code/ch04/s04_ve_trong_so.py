# Ô lệnh 4 -- Vẽ: màu = độ khó r_i, kích thước = số điểm g_i được sinh
X_min = X_toy[y_toy == 1]
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.scatter(*X_toy[y_toy == 0].T, s=8, color="#c9c8c2")
sc = ax.scatter(*X_min.T, c=r, cmap="Oranges", vmin=0, vmax=1, s=10 + g * 1.5,
                edgecolor="#52514e", linewidth=0.4)
fig.colorbar(sc, ax=ax, label="r_i (tỷ lệ láng giềng đa số)", shrink=0.85)
plt.savefig("figures/ch04_s04_weights.pdf", bbox_inches="tight")
