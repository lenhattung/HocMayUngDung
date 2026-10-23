# Ô lệnh 2 -- Đo "độ khó" r_i của một điểm thiểu số bằng 5 láng giềng
K = 5
nn_all = NearestNeighbors(n_neighbors=K + 1).fit(X_toy)     # tìm trong TOÀN BỘ dữ liệu
min_idx = np.where(y_toy == 1)[0]

for i in min_idx[[1, 32, 21]]:                              # ba điểm đã gặp ở Chương 3
    nb = nn_all.kneighbors(X_toy[[i]], return_distance=False)[0][1:]
    delta = (y_toy[nb] == 0).sum()                          # số láng giềng thuộc lớp đa số
    print(f"Điểm {i:3d}: nhãn láng giềng = {y_toy[nb]} -> Delta = {delta}, r = {delta / K:.1f}")
