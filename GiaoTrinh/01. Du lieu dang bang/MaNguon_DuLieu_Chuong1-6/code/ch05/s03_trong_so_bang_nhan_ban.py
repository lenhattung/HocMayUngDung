# Ô lệnh 3 -- Kiểm chứng: trọng số 3 cho một mẫu = nhân bản mẫu đó thành 3 bản
Xs = StandardScaler().fit_transform(X_train)[:2000]      # lấy 2 000 dòng cho nhanh
ys = y_train.values[:2000]

# Cách A: trọng số 3 cho mọi khách hàng vỡ nợ
lr_a = LogisticRegression(max_iter=5000).fit(Xs, ys, sample_weight=np.where(ys == 1, 3.0, 1.0))
# Cách B: chép mỗi khách hàng vỡ nợ thành 3 bản, không dùng trọng số
X_dup = np.vstack([Xs, Xs[ys == 1], Xs[ys == 1]])
y_dup = np.concatenate([ys, ys[ys == 1], ys[ys == 1]])
lr_b = LogisticRegression(max_iter=5000).fit(X_dup, y_dup)

print("5 hệ số đầu -- cách A:", lr_a.coef_[0][:5].round(4))
print("5 hệ số đầu -- cách B:", lr_b.coef_[0][:5].round(4))
print("Chênh lệch lớn nhất giữa hai cách:", f"{np.abs(lr_a.coef_ - lr_b.coef_).max():.1e}")
