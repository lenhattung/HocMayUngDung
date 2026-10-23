# Ô lệnh 16 -- Ma trận nhầm lẫn của hồi quy logistic
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_lr)
print(pd.DataFrame(cm, index=["Thực tế 0", "Thực tế 1"],
                   columns=["Dự đoán 0", "Dự đoán 1"]))

TN, FP, FN, TP = cm.ravel()       # "trải phẳng" ma trận 2x2 thành 4 con số
print(f"\nTN = {TN}, FP = {FP}, FN = {FN}, TP = {TP}")
