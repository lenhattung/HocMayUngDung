# Ô lệnh 8 -- Gộp các mã không có trong từ điển dữ liệu vào nhóm "khác"
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # 4 = khác
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # 3 = khác

for col in ["EDUCATION", "MARRIAGE"]:
    print(col, "->", df[col].value_counts().sort_index().to_dict())
