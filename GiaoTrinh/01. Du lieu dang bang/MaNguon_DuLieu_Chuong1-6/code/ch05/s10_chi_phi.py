# Ô lệnh 10 -- Chọn ngưỡng từ chi phí: t* = 1 / (1 + C_FN / C_FP)
def total_cost(p, t, c_fn, c_fp=1):
    pred = (p >= t).astype(int)
    fn = int(((pred == 0) & (y_test == 1)).sum())    # bỏ sót người vỡ nợ
    fp = int(((pred == 1) & (y_test == 0)).sum())    # báo động nhầm
    return fn, fp, fn * c_fn + fp * c_fp

rows = []
for c in [1, 3, 5, 10]:
    t_star = 1 / (1 + c)
    fn, fp, cost = total_cost(p1, t_star, c)
    rows.append({"C_FN/C_FP": c, "t*": round(t_star, 3), "FN": fn, "FP": fp,
                 "Chi phí (t*)": cost, "Chi phí (t = 0,5)": total_cost(p1, 0.5, c)[2]})
print(pd.DataFrame(rows).to_string(index=False))
