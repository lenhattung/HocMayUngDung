# Ô lệnh 3 -- Phân loại mọi điểm thiểu số thành SAFE / DANGER / NOISE
def minority_types(X, y, m=10):
    """Trả về loại của từng điểm thiểu số theo quy tắc của Han và cộng sự (2005)."""
    nn = NearestNeighbors(n_neighbors=m + 1).fit(X)
    nb = nn.kneighbors(X[y == 1], return_distance=False)[:, 1:]
    m_maj = (y[nb] == 0).sum(axis=1)
    types = np.where(m_maj == m, "NOISE",
                     np.where(m_maj >= m / 2, "DANGER", "SAFE"))
    return types, m_maj

types_toy, mmaj_toy = minority_types(X_toy, y_toy)
print(pd.Series(types_toy).value_counts().to_dict())
