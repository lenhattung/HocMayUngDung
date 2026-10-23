# Ô lệnh 2 -- Nạp thư viện và thiết lập chung cho cả chương
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 80)       # mỗi dòng bảng in ra tối đa 80 ký tự
pd.set_option("display.max_columns", 8)  # hiện tối đa 8 cột, phần còn lại thay bằng "..."
pd.set_option("display.precision", 3)    # làm tròn 3 chữ số sau dấu phẩy

SEED = 42  # "hạt giống" ngẫu nhiên: cố định để mỗi lần chạy cho cùng một kết quả
