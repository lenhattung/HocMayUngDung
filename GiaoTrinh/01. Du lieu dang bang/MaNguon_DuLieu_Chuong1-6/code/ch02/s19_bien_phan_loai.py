# Ô lệnh 19 -- SMOTE "phá hỏng" các biến phân loại
X_res_raw, _ = SMOTE(random_state=SEED).fit_resample(X_train, y_train)  # dữ liệu gốc cho dễ đọc
syn_raw = X_res_raw.iloc[len(X_train):]
print("Tỷ lệ mẫu tổng hợp có EDUCATION không phải số nguyên:",
      f"{(syn_raw['EDUCATION'] % 1 != 0).mean():.1%}")
print(syn_raw[["SEX", "EDUCATION", "MARRIAGE", "PAY_1"]].head(5).round(2))
