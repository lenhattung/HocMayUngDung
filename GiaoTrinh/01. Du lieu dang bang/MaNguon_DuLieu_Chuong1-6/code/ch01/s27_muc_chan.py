# Ô lệnh 27 -- Có bao nhiêu khu dân cư có giá bị "chặn trên"?
n_cap = (yh >= 5.0).sum()
print(f"Số khu có giá >= 5,0: {n_cap} ({n_cap / len(yh):.1%})")
print("Giá trị lớn nhất    :", yh.max())

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.hist(yh, bins=50, color="#2a78d6")
ax.set_xlabel("MedHouseVal (trăm nghìn USD)")
ax.set_ylabel("Số khu dân cư")
plt.savefig("figures/ch01_s27_hist.pdf", bbox_inches="tight")
