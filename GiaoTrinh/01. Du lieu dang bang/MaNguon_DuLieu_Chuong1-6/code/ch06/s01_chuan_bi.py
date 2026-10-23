# Ô lệnh 1 -- Nạp thư viện, dữ liệu và biến đổi đặc trưng cho mạng nơ-ron
import sys
sys.path.append("code")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from common import credit_split

pd.set_option("display.width", 80)
pd.set_option("display.precision", 3)
SEED = 42
torch.set_num_threads(2)

X_train, X_test, y_train, y_test = credit_split()
qt = QuantileTransformer(output_distribution="normal", random_state=0).fit(X_train)
Xtr, Xte = qt.transform(X_train), qt.transform(X_test)      # mảng NumPy đã biến đổi
print("PyTorch:", torch.__version__, "| Xtr:", Xtr.shape, "| Xte:", Xte.shape)
print("BILL_AMT1 trước biến đổi: min =", X_train["BILL_AMT1"].min(), ", max =", X_train["BILL_AMT1"].max())
print("BILL_AMT1 sau biến đổi  : min = %.2f, max = %.2f" % (Xtr[:, 11].min(), Xtr[:, 11].max()))
