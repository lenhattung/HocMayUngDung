# Ô lệnh 18 -- SMOTE làm "phình" xác suất dự đoán
print(f"Tỷ lệ vỡ nợ thật trên tập test        : {y_test.mean():.3f}")
print(f"Xác suất vỡ nợ TB -- XGBoost          : {proba_base.mean():.3f}")
print(f"Xác suất vỡ nợ TB -- XGBoost + SMOTE  : {p_smote.mean():.3f}")
