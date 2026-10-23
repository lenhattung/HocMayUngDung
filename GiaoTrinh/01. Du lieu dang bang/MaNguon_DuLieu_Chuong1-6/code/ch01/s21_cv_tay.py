# Ô lệnh 21 -- Đánh giá chéo 5-fold, viết vòng lặp "bằng tay" để thấy rõ từng bước
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = []
for k, (idx_tr, idx_va) in enumerate(skf.split(X_train, y_train), start=1):
    model = clone(logreg)                       # bản sao "mới tinh", chưa học gì
    model.fit(X_train.iloc[idx_tr], y_train.iloc[idx_tr])
    p = model.predict_proba(X_train.iloc[idx_va])[:, 1]
    s = average_precision_score(y_train.iloc[idx_va], p)
    scores.append(s)
    print(f"Fold {k}: học {len(idx_tr)} mẫu, kiểm tra {len(idx_va)} mẫu -> PR-AUC = {s:.4f}")

print(f"\nTrung bình ± độ lệch chuẩn: {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")
