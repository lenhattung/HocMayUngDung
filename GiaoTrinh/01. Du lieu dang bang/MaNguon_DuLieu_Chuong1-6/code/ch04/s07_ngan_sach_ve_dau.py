# Ô lệnh 7 -- Ngân sách đi về đâu? So sánh ba cách phân bổ (cùng dùng 5 láng giềng)
groups = pd.cut(r, bins=[-0.01, 0.49, 0.99, 1.0], labels=["r < 0,5", "0,5 <= r < 1", "r = 1"])
share = pd.DataFrame({
    "Số điểm": pd.Series(groups).value_counts().sort_index(),
    "SMOTE": pd.Series(np.ones(len(r)), index=groups).groupby(level=0, observed=False).sum() / len(r),
    "Kiểu Borderline": pd.Series(((r >= 0.5) & (r < 1)).astype(float), index=groups)
                    .groupby(level=0, observed=False).sum() / ((r >= 0.5) & (r < 1)).sum(),
    "ADASYN": pd.Series(g, index=groups).groupby(level=0, observed=False).sum() / g.sum(),
})
print(share.round(2))
