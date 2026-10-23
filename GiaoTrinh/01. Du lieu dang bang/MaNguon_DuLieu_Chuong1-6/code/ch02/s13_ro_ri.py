# Ô lệnh 13 -- Làm SAI: chạy SMOTE trên toàn bộ tập train RỒI MỚI chia fold
X_leak, y_leak = SMOTE(random_state=SEED).fit_resample(scaler.transform(X_train), y_train)
r_wrong = cross_validate(xgb_clf(), X_leak, y_leak, cv=skf, scoring=scoring)
wrong = {k: r_wrong["test_" + k].mean() for k in scoring}

print(pd.DataFrame({"ĐÚNG (SMOTE trong pipeline)": cv_table["XGBoost + SMOTE"],
                    "SAI (SMOTE trước khi chia)": wrong}).round(3))
