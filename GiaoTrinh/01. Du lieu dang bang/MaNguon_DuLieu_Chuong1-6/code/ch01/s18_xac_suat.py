# Ô lệnh 18 -- Xác suất vỡ nợ mà mô hình gán cho 8 khách hàng đầu tiên của tập test
proba_lr = logreg.predict_proba(X_test)[:, 1]   # cột 1 = xác suất thuộc lớp 1 (vỡ nợ)

view = pd.DataFrame({"Xác suất vỡ nợ": proba_lr[:8],
                     "Dự đoán (ngưỡng 0,5)": pred_lr[:8],
                     "Thực tế": y_test.values[:8]})
print(view)
