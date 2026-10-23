# Ô lệnh 28 -- Tính MAE, RMSE và R² bằng tay trên ví dụ 4 căn nhà
y_true = np.array([2.0, 3.0, 1.5, 4.0])     # giá thật
y_pred = np.array([2.5, 2.5, 1.5, 3.0])     # giá mô hình dự đoán
err = y_true - y_pred

print("Sai số:", err)
print("MAE  =", np.mean(np.abs(err)))
print("RMSE =", np.sqrt(np.mean(err ** 2)).round(4))
print("R²   =", (1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)).round(4))
