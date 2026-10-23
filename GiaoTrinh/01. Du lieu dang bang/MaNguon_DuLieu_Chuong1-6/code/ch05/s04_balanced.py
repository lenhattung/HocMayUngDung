# Ô lệnh 4 -- Trọng số "balanced" của scikit-learn
from sklearn.utils.class_weight import compute_class_weight

w_bal = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
ratio = w_bal[1] / w_bal[0]
print(f"w0 = {w_bal[0]:.3f}, w1 = {w_bal[1]:.3f}, tỷ số w1/w0 = {ratio:.3f}")
print(f"Tổng trọng số lớp 0: {w_bal[0] * n0:.0f} | lớp 1: {w_bal[1] * n1:.0f}")
