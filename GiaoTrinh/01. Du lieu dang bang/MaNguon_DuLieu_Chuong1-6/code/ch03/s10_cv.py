# Ô lệnh 10 -- So sánh 4 chiến lược lấy mẫu với 2 mô hình (đánh giá chéo, ~1 phút)
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"PR-AUC": "average_precision", "F1": "f1", "Recall": "recall", "Precision": "precision"}
strategies = {"Không lấy mẫu": None, **samplers}

def make(model, sampler):
    steps = [("scale", StandardScaler())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    return Pipeline(steps + [("model", model)])

cv_rows = {}
for mname, mk in {"LogReg": lambda: LogisticRegression(max_iter=2000), "XGBoost": xgb_clf}.items():
    for sname, s in strategies.items():
        r = cross_validate(make(mk(), s), X_train, y_train, cv=skf, scoring=scoring)
        cv_rows[f"{mname} | {sname}"] = {k: r["test_" + k].mean() for k in scoring}
cv_df = pd.DataFrame(cv_rows).T
print(cv_df)
