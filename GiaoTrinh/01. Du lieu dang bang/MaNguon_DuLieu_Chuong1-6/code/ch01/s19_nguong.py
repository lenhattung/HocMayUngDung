# Ô lệnh 19 -- Cùng một mô hình, đổi ngưỡng quyết định thì kết quả thay đổi thế nào?
rows = []
for t in [0.5, 0.4, 0.3, 0.2]:
    pred_t = (proba_lr >= t).astype(int)          # 1 nếu xác suất >= ngưỡng t
    rows.append({"Ngưỡng": t,
                 "Số bị gắn cờ": pred_t.sum(),
                 "Precision": precision_score(y_test, pred_t),
                 "Recall": recall_score(y_test, pred_t),
                 "F1": f1_score(y_test, pred_t)})
print(pd.DataFrame(rows).to_string(index=False))
