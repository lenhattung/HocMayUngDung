# Ô lệnh 5 -- Tự viết ADASYN: mỗi điểm sinh đúng g_i điểm mới
def adasyn_tay(X_min, g, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)
    new = []
    for i in range(len(X_min)):                  # KHÁC SMOTE: duyệt từng điểm,
        for _ in range(g[i]):                    # mỗi điểm sinh đúng g_i lần
            j = rng.choice(idx[i][1:])
            lam = rng.uniform(0, 1)
            new.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    return np.array(new)

X_syn = adasyn_tay(X_min, g)
print("Số điểm sinh ra:", len(X_syn))
top = np.argsort(-g)[:3]
print("Ba điểm sinh nhiều nhất:", {int(i): int(g[i]) for i in top},
      "-> chiếm", f"{g[top].sum() / g.sum():.0%}", "tổng số điểm mới")
