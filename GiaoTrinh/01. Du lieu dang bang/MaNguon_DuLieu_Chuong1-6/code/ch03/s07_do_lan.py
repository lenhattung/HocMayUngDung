# Ô lệnh 7 -- Đo xem điểm tổng hợp "lấn" vào vùng của lớp đa số đến mức nào
nn10 = NearestNeighbors(n_neighbors=10).fit(X_toy)       # chỉ dùng dữ liệu GỐC
for name, S_ in synth.items():
    nb = nn10.kneighbors(S_, return_distance=False)
    share_maj = (y_toy[nb] == 0).mean(axis=1)              # tỷ lệ láng giềng thuộc lớp đa số
    print(f"{name:13s}: TB {share_maj.mean():.0%} láng giềng là lớp đa số;"
          f" {np.mean(share_maj > 0.5):.0%} điểm nằm hẳn trong vùng đa số")
