# Ô lệnh 10 -- SMOTE thay đổi tập train như thế nào?
X_res, y_res = SMOTE(random_state=SEED).fit_resample(scaler.transform(X_train), y_train)

print("Trước SMOTE:", y_train.value_counts().to_dict())
print("Sau SMOTE  :", pd.Series(y_res).value_counts().to_dict())
print("Số mẫu tổng hợp được thêm vào:", len(y_res) - len(y_train))
print("Tập test vẫn giữ nguyên       :", y_test.value_counts().to_dict())
