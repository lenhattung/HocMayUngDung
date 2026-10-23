# Ô lệnh 2 -- Hàm mất mát log loss, có và không có trọng số, tính tay cho 4 khách hàng
y_true = np.array([1, 1, 0, 0])
p_hat = np.array([0.30, 0.80, 0.20, 0.60])     # xác suất vỡ nợ mà mô hình dự đoán
w = 3.5                                        # trọng số cho lớp vỡ nợ (lớp 1)

loss = -(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat))   # mất mát từng người
weight = np.where(y_true == 1, w, 1.0)
print(pd.DataFrame({"y": y_true, "p_hat": p_hat, "mất mát": loss.round(3),
                    "trọng số": weight, "mất mát x trọng số": (loss * weight).round(3)}))
print(f"\nTổng mất mát -- không trọng số: {loss.sum():.3f} | có trọng số: {(loss * weight).sum():.3f}")
