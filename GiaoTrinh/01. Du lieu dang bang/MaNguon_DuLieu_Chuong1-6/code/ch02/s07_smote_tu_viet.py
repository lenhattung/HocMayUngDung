# Ô lệnh 7 -- Tự viết hàm SMOTE hoàn chỉnh (khoảng 10 dòng)
def smote_tay(X_min, G, k=5, seed=SEED):
    """Sinh G điểm tổng hợp từ tập thiểu số X_min (phiên bản tối giản để học)."""
    rng = np.random.default_rng(seed)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)
    new = []
    for _ in range(G):
        i = rng.integers(len(X_min))          # chọn ngẫu nhiên một điểm thiểu số
        j = rng.choice(idx[i][1:])            # chọn một trong k láng giềng của nó
        lam = rng.uniform(0, 1)
        new.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    return np.array(new)

G = (y_toy == 0).sum() - (y_toy == 1).sum()   # số điểm cần sinh để hai lớp bằng nhau
X_syn = smote_tay(X_min, G)
print("Cần sinh:", G, "điểm | Kích thước kết quả:", X_syn.shape)
