# Ô lệnh 5 -- Bỏ cột ID và đặt lại tên cột cho dễ dùng
df = df.drop(columns="ID")
df = df.rename(columns={"PAY_0": "PAY_1",
                        "default payment next month": "default"})

cols = df.columns.tolist()
for i in range(0, len(cols), 6):      # in 6 tên cột trên mỗi dòng cho gọn
    print(cols[i:i + 6])
