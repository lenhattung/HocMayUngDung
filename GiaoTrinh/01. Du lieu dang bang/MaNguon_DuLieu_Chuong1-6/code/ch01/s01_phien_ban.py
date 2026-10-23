# Ô lệnh 1 -- Kiểm tra phiên bản Python và các thư viện chính
import sys
import numpy, pandas, sklearn, matplotlib, xgboost, lightgbm

print("Python       :", sys.version.split()[0])
print("numpy        :", numpy.__version__)
print("pandas       :", pandas.__version__)
print("scikit-learn :", sklearn.__version__)
print("matplotlib   :", matplotlib.__version__)
print("xgboost      :", xgboost.__version__)
print("lightgbm     :", lightgbm.__version__)
