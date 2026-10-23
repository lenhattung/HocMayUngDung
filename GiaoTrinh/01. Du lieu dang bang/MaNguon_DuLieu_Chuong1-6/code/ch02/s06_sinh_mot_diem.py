# Ô lệnh 6 -- Bước 2 của SMOTE: sinh MỘT điểm mới bằng tay
rng = np.random.default_rng(SEED)
j = rng.choice(idx[0][1:])           # chọn ngẫu nhiên 1 trong 5 láng giềng
lam = rng.uniform(0, 1)              # hệ số lambda ngẫu nhiên trong khoảng (0, 1)

x_i, x_nb = X_min[0], X_min[j]
x_new = x_i + lam * (x_nb - x_i)     # công thức SMOTE

print(f"Láng giềng được chọn: điểm số {j}, lambda = {lam:.3f}")
print("x_i   =", x_i.round(3))
print("x_nb  =", x_nb.round(3))
print("x_new =", x_new.round(3))
