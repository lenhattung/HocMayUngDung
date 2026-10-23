# Ô lệnh 3 -- Vẽ focal loss theo p_t với các giá trị gamma khác nhau
pt = np.linspace(0.01, 1, 300)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3))
for g, c in zip([0, 0.5, 1, 2, 5], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]):
    ax[0].plot(pt, -(1 - pt) ** g * np.log(pt), color=c, label=f"gamma = {g}")
    ax[1].plot(pt, (1 - pt) ** g, color=c)
ax[0].set_ylim(0, 4); ax[0].set_xlabel("p_t"); ax[0].set_ylabel("Mất mát"); ax[0].legend(fontsize=7)
ax[0].set_title("Focal loss (gamma = 0 là cross-entropy)")
ax[1].set_xlabel("p_t"); ax[1].set_ylabel("(1 - p_t)^gamma"); ax[1].set_title("Hệ số giảm trọng số")
plt.savefig("figures/ch06_s03_curves.pdf", bbox_inches="tight")
