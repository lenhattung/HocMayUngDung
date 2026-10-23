# Ô lệnh 20 -- SMOTE-NC: phiên bản dành cho dữ liệu có biến phân loại
from imblearn.over_sampling import SMOTENC

cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
smnc = SMOTENC(categorical_features=[X_train.columns.get_loc(c) for c in cat_cols],
               random_state=SEED)
X_nc, y_nc = smnc.fit_resample(X_train, y_train)
syn_nc = X_nc.iloc[len(X_train):]
print("Tỷ lệ mẫu tổng hợp có EDUCATION không phải số nguyên:",
      f"{(syn_nc['EDUCATION'] % 1 != 0).mean():.1%}")
print(syn_nc[cat_cols + ["PAY_1"]].head(5).round(2))
