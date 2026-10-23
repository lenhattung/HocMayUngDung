# Ô lệnh 29 -- Chia dữ liệu và so sánh ba mô hình hồi quy bằng 5-fold CV
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    Xh, yh, test_size=0.2, random_state=SEED)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

reg_models = {
    "Dummy (trung bình)": DummyRegressor(strategy="mean"),
    "LinearReg": make_pipeline(StandardScaler(), LinearRegression()),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=2, random_state=SEED, verbose=-1),
}
reg_scoring = {"RMSE": "neg_root_mean_squared_error",
               "MAE": "neg_mean_absolute_error", "R2": "r2"}
tab = {}
for name, m in reg_models.items():
    r = cross_validate(m, Xh_train, yh_train, cv=kf, scoring=reg_scoring)
    tab[name] = {"RMSE": -r["test_RMSE"].mean(),   # đổi dấu: scikit-learn trả về số âm
                 "MAE": -r["test_MAE"].mean(),
                 "R2": r["test_R2"].mean()}
print(pd.DataFrame(tab).T.round(3))
