# Ô lệnh 1 -- Nạp thư viện và dữ liệu đã chia ở Chương 1
import sys
sys.path.append("code")              # để Python tìm thấy file code/common.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import credit_split, xgb_clf

pd.set_option("display.width", 80)
pd.set_option("display.precision", 3)
SEED = 42

X_train, X_test, y_train, y_test = credit_split()     # đúng cách chia của Bước 13, Chương 1
print("Train:", X_train.shape, "  Test:", X_test.shape)
print("Số khách hàng trong tập train theo nhãn:", y_train.value_counts().to_dict())
