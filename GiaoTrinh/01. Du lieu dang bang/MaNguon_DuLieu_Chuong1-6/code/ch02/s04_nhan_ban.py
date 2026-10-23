# Ô lệnh 4 -- Cách đơn giản nhất: nhân bản ngẫu nhiên các điểm thiểu số
from imblearn.over_sampling import RandomOverSampler

X_ros, y_ros = RandomOverSampler(random_state=SEED).fit_resample(X_toy, y_toy)
print("Sau khi nhân bản [lớp 0, lớp 1]:", np.bincount(y_ros))

minority = X_ros[y_ros == 1]
print("Số điểm lớp 1:", len(minority),
      "| trong đó số điểm KHÁC NHAU:", len(np.unique(minority, axis=0)))
