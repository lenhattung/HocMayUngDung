# Ô lệnh 2 -- "Hỏi" 10 láng giềng của một điểm thiểu số xem chúng thuộc lớp nào
m = 10
nn_all = NearestNeighbors(n_neighbors=m + 1).fit(X_toy)    # tìm trong TOÀN BỘ dữ liệu
min_idx = np.where(y_toy == 1)[0]                          # vị trí 66 điểm thiểu số

for i in min_idx[[1, 32, 21]]:                             # xét 3 điểm làm ví dụ
    nb = nn_all.kneighbors(X_toy[[i]], return_distance=False)[0][1:]   # bỏ chính nó
    m_maj = (y_toy[nb] == 0).sum()                         # số láng giềng thuộc lớp đa số
    print(f"Điểm {i:3d}: nhãn láng giềng = {y_toy[nb]} -> {m_maj}/10 thuộc lớp đa số")
