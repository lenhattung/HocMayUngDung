# Ô lệnh 30 -- Đánh giá LightGBM trên tập test và xem vài dự đoán cụ thể
from sklearn.metrics import root_mean_squared_error, r2_score

lgb_reg = reg_models["LightGBM"].fit(Xh_train, yh_train)
pred_h = lgb_reg.predict(Xh_test)
print(f"Test RMSE = {root_mean_squared_error(yh_test, pred_h):.3f},"
      f"  R2 = {r2_score(yh_test, pred_h):.3f}")
print()
print(pd.DataFrame({"Giá thật": yh_test.values[:6], "Dự đoán": pred_h[:6],
                    "Sai số": yh_test.values[:6] - pred_h[:6]}).round(3))
