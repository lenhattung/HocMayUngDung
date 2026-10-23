# Ô lệnh 3 -- Chia "ngân sách" G điểm tổng hợp cho từng điểm thiểu số
def adasyn_weights(X, y, K=5):
    """Tính r_i (độ khó), r_hat_i (đã chuẩn hóa) và g_i (số điểm sẽ sinh từ điểm i)."""
    G = (y == 0).sum() - (y == 1).sum()                     # tổng số điểm cần sinh
    nb = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(
        X[y == 1], return_distance=False)[:, 1:]
    r = (y[nb] == 0).sum(axis=1) / K
    r_hat = r / r.sum()                                     # tổng các r_hat bằng 1
    g = np.rint(r_hat * G).astype(int)                      # làm tròn thành số nguyên
    return r, r_hat, g, G

r, r_hat, g, G = adasyn_weights(X_toy, y_toy)
print(f"G = {G}, tổng các r_i = {r.sum():.1f}, tổng các g_i = {g.sum()}")
print(pd.DataFrame({"r_i": r, "r_hat_i": r_hat, "g_i": g}).iloc[[1, 32, 21]])
print("Số điểm có g_i = 0:", (g == 0).sum(), "/", len(g), "| g_i lớn nhất:", g.max())
