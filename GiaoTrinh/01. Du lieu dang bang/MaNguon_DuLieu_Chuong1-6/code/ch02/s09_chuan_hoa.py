# Ô lệnh 9 -- Vì sao phải chuẩn hóa trước SMOTE? So sánh "láng giềng gần nhất"
from sklearn.preprocessing import StandardScaler

X_min_c = X_train[y_train == 1]                     # 5 309 khách hàng vỡ nợ (tập train)
scaler = StandardScaler().fit(X_train)
nb_raw = NearestNeighbors(n_neighbors=2).fit(X_min_c).kneighbors(X_min_c)[1][:, 1]
Z = scaler.transform(X_min_c)
nb_std = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[1][:, 1]

i = 2                                               # một ví dụ cụ thể
cols = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
print(pd.DataFrame({"Khách hàng": X_min_c.iloc[i][cols],
                    "LG chưa chuẩn hóa": X_min_c.iloc[nb_raw[i]][cols],
                    "LG đã chuẩn hóa": X_min_c.iloc[nb_std[i]][cols]}))

same = lambda nb: (X_min_c["PAY_1"].values == X_min_c["PAY_1"].values[nb]).mean()
print(f"\nTỷ lệ láng giềng có CÙNG PAY_1 -- chưa chuẩn hóa: {same(nb_raw):.1%},"
      f" đã chuẩn hóa: {same(nb_std):.1%}")
