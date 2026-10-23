# Ô lệnh 4 -- Vẽ ba loại điểm thiểu số lên mặt phẳng
X_min = X_toy[y_toy == 1]
style = {"SAFE": ("#1baf7a", "o"), "DANGER": ("#eb6834", "D"), "NOISE": ("#e34948", "s")}

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.scatter(*X_toy[y_toy == 0].T, s=8, color="#c9c8c2", label="Lớp đa số")
for t, (c, mk) in style.items():
    ax.scatter(*X_min[types_toy == t].T, s=22, color=c, marker=mk, edgecolor="white",
               linewidth=0.4, label=f"{t} ({(types_toy == t).sum()})")
ax.legend(fontsize=7, loc="lower right")
plt.savefig("figures/ch03_s04_types.pdf", bbox_inches="tight")
