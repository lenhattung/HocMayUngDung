# Ô lệnh 2 -- Tính tay focal loss cho 4 khách hàng
y = np.array([1, 1, 0, 0])
p = np.array([0.95, 0.30, 0.10, 0.70])          # xác suất vỡ nợ mô hình dự đoán
p_t = np.where(y == 1, p, 1 - p)                # xác suất gán cho NHÃN ĐÚNG
ce = -np.log(p_t)                               # cross-entropy (log loss)

tab = pd.DataFrame({"y": y, "p": p, "p_t": p_t, "CE": ce})
for g in [1, 2]:
    tab[f"(1-p_t)^{g}"] = (1 - p_t) ** g
    tab[f"FL g={g}"] = (1 - p_t) ** g * ce
print(tab.round(3).to_string())
