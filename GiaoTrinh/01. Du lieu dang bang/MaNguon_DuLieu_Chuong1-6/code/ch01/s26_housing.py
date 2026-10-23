# Ô lệnh 26 -- Bộ dữ liệu thứ hai: California Housing (có sẵn trong scikit-learn)
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)   # lần đầu sẽ tự tải về máy
Xh, yh = housing.data, housing.target

print("Kích thước:", Xh.shape)
print(Xh.head(3).round(2).T)        # .T: xoay bảng cho gọn
print()
print(yh.describe().round(3))
