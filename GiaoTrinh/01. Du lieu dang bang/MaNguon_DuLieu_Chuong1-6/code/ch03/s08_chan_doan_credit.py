# Ô lệnh 8 -- Chẩn đoán dữ liệu thật: khách hàng vỡ nợ thuộc loại nào?
from sklearn.preprocessing import StandardScaler

Z_train = StandardScaler().fit_transform(X_train)          # chuẩn hóa trước khi tính khoảng cách
types_c, mmaj_c = minority_types(Z_train, y_train.values)
ratio = pd.Series(types_c).value_counts(normalize=True)
print("Credit Default:", (ratio * 100).round(1).to_dict(), "(%)")
toy_ratio = pd.Series(types_toy).value_counts(normalize=True)
print("Dữ liệu đồ chơi:", (toy_ratio * 100).round(1).to_dict(), "(%)")
