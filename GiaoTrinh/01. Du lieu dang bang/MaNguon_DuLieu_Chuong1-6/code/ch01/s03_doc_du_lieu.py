# Ô lệnh 3 -- Đọc file Excel vào một bảng dữ liệu (DataFrame)
df = pd.read_excel("data/credit_default.xls", header=1)

print("Số dòng, số cột:", df.shape)
print()
print(df.iloc[:5, :7])   # 5 dòng đầu, 7 cột đầu
