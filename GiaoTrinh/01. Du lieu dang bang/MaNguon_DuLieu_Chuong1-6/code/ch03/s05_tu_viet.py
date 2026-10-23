# Ô lệnh 5 -- Tự viết Borderline-SMOTE-1: chỉ sinh mẫu từ các điểm DANGER
def borderline_smote_tay(X_min, types, G, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)
    danger = np.where(types == "DANGER")[0]          # chỉ những điểm này được làm "hạt giống"
    new = []
    for _ in range(G):
        i = rng.choice(danger)                       # KHÁC SMOTE: chỉ chọn trong DANGER
        j = rng.choice(idx[i][1:])                   # láng giềng vẫn là mọi điểm thiểu số
        lam = rng.uniform(0, 1)
        new.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    return np.array(new)

G = (y_toy == 0).sum() - (y_toy == 1).sum()
X_syn_b1 = borderline_smote_tay(X_min, types_toy, G)
print("Số điểm DANGER làm hạt giống:", (types_toy == "DANGER").sum(), "| số điểm sinh ra:", len(X_syn_b1))
