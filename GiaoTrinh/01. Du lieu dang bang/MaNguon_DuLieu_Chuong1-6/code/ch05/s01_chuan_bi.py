# Ô lệnh 1 -- Nạp thư viện và dữ liệu
import sys
sys.path.append("code")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import credit_split, xgb_clf

pd.set_option("display.width", 80)
pd.set_option("display.precision", 3)
SEED = 42

X_train, X_test, y_train, y_test = credit_split()
n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
print(f"Tập train: {n0} không vỡ nợ, {n1} vỡ nợ, tỷ số n0/n1 = {n0 / n1:.2f}")
