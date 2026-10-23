# Ô lệnh 14 -- Vì sao làm sai lại cho điểm cao như vậy? Đo khoảng cách
real_min = Z                                  # 5 309 khách hàng vỡ nợ thật (đã chuẩn hóa)
synth = X_leak[len(X_train):]                 # 13 382 điểm tổng hợp (nằm ở cuối mảng)
nn_real = NearestNeighbors(n_neighbors=2).fit(real_min)

d_real = nn_real.kneighbors(real_min)[0][:, 1]   # điểm thật -> điểm thật KHÁC gần nhất
d_syn = nn_real.kneighbors(synth)[0][:, 0]       # điểm tổng hợp -> điểm thật gần nhất
print(f"Trung vị khoảng cách: điểm thật     -> điểm thật khác gần nhất: {np.median(d_real):.3f}")
print(f"Trung vị khoảng cách: điểm tổng hợp -> điểm thật gần nhất     : {np.median(d_syn):.3f}")
