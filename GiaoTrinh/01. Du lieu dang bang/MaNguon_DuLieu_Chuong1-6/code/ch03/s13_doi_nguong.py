# Ô lệnh 13 -- Đối chiếu với phương án dời ngưỡng của Chương 2 (t* = 0,26)
t_star = 0.26                                   # ngưỡng tìm được bằng out-of-fold ở Chương 2
pred = (proba["Không lấy mẫu"] >= t_star).astype(int)
print(f"XGBoost, t = 0,26 -> F1 = {f1_score(y_test, pred):.3f}, "
      f"Recall = {recall_score(y_test, pred):.3f}, Precision = {precision_score(y_test, pred):.3f}")
