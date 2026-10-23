# Ô lệnh 9 -- Đánh giá chéo: không lấy mẫu, SMOTE, ADASYN (2 mô hình)
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"PR-AUC": "average_precision", "F1": "f1", "Recall": "recall", "Precision": "precision"}

def make(model, sampler):
    steps = [("scale", StandardScaler())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    return Pipeline(steps + [("model", model)])

strategies = {"Không lấy mẫu": None, "SMOTE": samplers["SMOTE"], "ADASYN": samplers["ADASYN"]}
cv_rows = {}
for mname, mk in {"LogReg": lambda: LogisticRegression(max_iter=2000), "XGBoost": xgb_clf}.items():
    for sname, s in strategies.items():
        r_ = cross_validate(make(mk(), s), X_train, y_train, cv=skf, scoring=scoring)
        cv_rows[f"{mname} | {sname}"] = {k: r_["test_" + k].mean() for k in scoring}
print(pd.DataFrame(cv_rows).T)
