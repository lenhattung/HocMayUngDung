# Ô lệnh 23 -- So sánh bốn mô hình bằng cùng một quy trình đánh giá chéo
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    "LogReg": logreg,
    "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                           n_jobs=2, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=2, random_state=SEED),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               n_jobs=2, random_state=SEED, verbose=-1),
}
table = {}
for name, m in models.items():
    r = cross_validate(m, X_train, y_train, cv=skf, scoring=scoring)
    table[name] = {k: f"{r['test_' + k].mean():.3f} ± {r['test_' + k].std(ddof=1):.3f}"
                   for k in scoring}
print(pd.DataFrame(table).T)
