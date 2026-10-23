# Ô lệnh 4 -- Kiểm tra kiểu dữ liệu, giá trị thiếu và dòng trùng lặp
print("Kiểu dữ liệu của các cột:")
print(df.dtypes.value_counts())
print()
print("Tổng số ô bị thiếu (NaN):", df.isna().sum().sum())
print("Số dòng trùng lặp (bỏ qua cột ID):", df.drop(columns="ID").duplicated().sum())
