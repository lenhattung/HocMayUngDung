# Ô lệnh 3 -- Một bộ dữ liệu "đồ chơi" 2 chiều để nhìn thấy SMOTE bằng mắt
from sklearn.datasets import make_classification

X_toy, y_toy = make_classification(n_samples=600, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.9, 0.1],
                                   class_sep=1.0, random_state=3)
print("Số điểm mỗi lớp [lớp 0, lớp 1]:", np.bincount(y_toy))
print("5 điểm đầu tiên:")
print(np.round(X_toy[:5], 3))
