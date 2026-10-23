# Ô lệnh 11 -- So sánh có/không SMOTE bằng đánh giá chéo 5-fold
from imblearn.pipeline import Pipeline          # CHÚ Ý: Pipeline của imblearn, không phải sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
           "F1": "f1", "Recall": "recall", "Precision": "precision"}

def make(model, smote):
    """Ghép chuẩn hóa -> (SMOTE) -> mô hình thành một pipeline."""
    steps = [("scale", StandardScaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    return Pipeline(steps + [("model", model)])

configs = {"LogReg": make(LogisticRegression(max_iter=2000), False),
           "LogReg + SMOTE": make(LogisticRegression(max_iter=2000), True),
           "XGBoost": make(xgb_clf(), False),
           "XGBoost + SMOTE": make(xgb_clf(), True)}
cv_table = {}
for name, pipe in configs.items():
    r = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring)
    cv_table[name] = {k: r["test_" + k].mean() for k in scoring}
print(pd.DataFrame(cv_table).T)
