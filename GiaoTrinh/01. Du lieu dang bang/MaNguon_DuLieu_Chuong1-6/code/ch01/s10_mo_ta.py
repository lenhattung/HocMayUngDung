# Ô lệnh 10 -- Thống kê mô tả cho bốn biến số tiêu biểu
num_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
desc = df[num_cols].describe().T            # .T: xoay bảng để mỗi biến là một dòng
print(desc[["mean", "std", "min", "50%", "max"]].round(0))
