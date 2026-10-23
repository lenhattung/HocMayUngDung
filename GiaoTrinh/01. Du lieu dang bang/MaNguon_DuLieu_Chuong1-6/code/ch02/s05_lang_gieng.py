# Ô lệnh 5 -- Bước 1 của SMOTE: tìm 5 láng giềng gần nhất (cùng lớp thiểu số)
from sklearn.neighbors import NearestNeighbors

X_min = X_toy[y_toy == 1]                         # 66 điểm thiểu số
nn = NearestNeighbors(n_neighbors=6).fit(X_min)   # 6 = chính nó + 5 láng giềng
dist, idx = nn.kneighbors(X_min[[0]])             # hỏi láng giềng của điểm số 0

print("Điểm x_0          :", X_min[0].round(3))
print("Chỉ số 5 láng giềng:", idx[0][1:])           # bỏ phần tử đầu (chính nó)
print("Khoảng cách        :", dist[0][1:].round(3))
