# Ô lệnh 9 -- Trạng thái trả nợ tháng 9 (PAY_1) nhận những giá trị nào?
print(df["PAY_1"].value_counts().sort_index().to_dict())
