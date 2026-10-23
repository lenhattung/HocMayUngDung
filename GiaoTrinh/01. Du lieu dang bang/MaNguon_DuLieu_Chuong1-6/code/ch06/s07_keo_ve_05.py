# Ô lệnh 7 -- Nếu xác suất thật là p, focal loss "muốn" dự đoán bao nhiêu?
q = np.linspace(1e-4, 1 - 1e-4, 200001)
rows = {}
for g in [0, 1, 2, 5]:
    rows[f"gamma = {g}"] = [q[np.argmin(-p_ * (1 - q) ** g * np.log(q) - (1 - p_) * q ** g * np.log(1 - q))]
                            for p_ in [0.05, 0.2, 0.4, 0.6, 0.9]]
print(pd.DataFrame(rows, index=["p = 0,05", "p = 0,2", "p = 0,4", "p = 0,6", "p = 0,9"]).T.round(3))
