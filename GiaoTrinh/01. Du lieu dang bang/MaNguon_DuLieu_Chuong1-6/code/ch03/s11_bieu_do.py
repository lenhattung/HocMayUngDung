# Ô lệnh 11 -- Biểu đồ cột cho XGBoost: xu hướng khi càng "tập trung vào biên"
xg = cv_df[cv_df.index.str.startswith("XGBoost")]
xg.index = [i.split("| ")[1] for i in xg.index]

ax = xg.T.plot(kind="bar", figsize=(7, 2.8), width=0.8, rot=0,
               color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
ax.set_ylim(0.3, 0.72); ax.set_ylabel("Điểm CV trung bình")
ax.legend(ncol=4, fontsize=7, loc="upper center")
plt.savefig("figures/ch03_s11_bars.pdf", bbox_inches="tight")
