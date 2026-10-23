# Ô lệnh 1 -- Nạp thư viện, dữ liệu Credit Default và bộ dữ liệu "đồ chơi"
import sys
sys.path.append("code")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from common import credit_split, xgb_clf

pd.set_option("display.width", 80)
pd.set_option("display.precision", 3)
SEED = 42

X_train, X_test, y_train, y_test = credit_split()
X_toy, y_toy = make_classification(n_samples=600, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.9, 0.1],
                                   class_sep=1.0, random_state=3)
print("Credit -- train:", X_train.shape, "| Đồ chơi:", X_toy.shape, np.bincount(y_toy))
