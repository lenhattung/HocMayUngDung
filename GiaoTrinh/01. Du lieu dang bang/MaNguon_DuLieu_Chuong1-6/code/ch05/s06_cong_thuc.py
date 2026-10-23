# Ô lệnh 6 -- Trọng số w "uốn" thang xác suất theo công thức p_w = w p / (w p + 1 - p)
p = np.array([0.05, 0.10, 0.221, 0.30, 0.50, 0.80])
w_ = ratio
p_w = w_ * p / (w_ * p + 1 - p)
print(pd.DataFrame({"p (không trọng số)": p, f"p_w (w = {w_:.2f})": p_w.round(3)}).to_string(index=False))
print(f"\nNgưỡng tương đương: p_w >= 0,5  <=>  p >= 1/(1+w) = {1 / (1 + w_):.3f}")
