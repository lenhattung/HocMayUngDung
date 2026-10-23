# Ô lệnh 9 -- Cái giá phải trả: xác suất bị méo
print(f"Tỷ lệ vỡ nợ thật trên tập test : {y_test.mean():.3f}")
print(f"Xác suất TB -- w = 1           : {p1.mean():.3f}")
print(f"Xác suất TB -- w = {ratio:.2f}        : {pw.mean():.3f}")

# Khử trọng số: đảo ngược công thức ở Ô lệnh 6
p_back = pw / (pw + ratio * (1 - pw))
print(f"Xác suất TB sau khi khử trọng số: {p_back.mean():.3f}")
