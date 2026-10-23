# Ô lệnh 7 -- Ba biến phân loại nhận những giá trị nào?
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(col, "->", df[col].value_counts().sort_index().to_dict())
