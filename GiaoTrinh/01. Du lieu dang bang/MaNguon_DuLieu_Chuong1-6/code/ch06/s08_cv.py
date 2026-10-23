# Ô lệnh 8 -- So sánh 8 hàm mất mát bằng đánh giá chéo 5-fold (khoảng 2--3 phút)
from sklearn.model_selection import StratifiedKFold

w_bal = (y_train == 0).sum() / (y_train == 1).sum()
losses = {
    "BCE": bce,
    "BCE có trọng số": lambda z, y: F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(w_bal)),
    "FL gamma=0.5": lambda z, y: focal_loss(z, y, 0.5),
    "FL gamma=1": lambda z, y: focal_loss(z, y, 1.0),
    "FL gamma=2": lambda z, y: focal_loss(z, y, 2.0),
    "FL gamma=5": lambda z, y: focal_loss(z, y, 5.0),
    "FL gamma=2, alpha=0.25": lambda z, y: focal_loss(z, y, 2.0, 0.25),
    "FL gamma=2, alpha=0.75": lambda z, y: focal_loss(z, y, 2.0, 0.75),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = {}
for name, loss_fn in losses.items():
    folds = []
    for tr, va in skf.split(X_train, y_train):
        qt_f = QuantileTransformer(output_distribution="normal", random_state=0).fit(X_train.iloc[tr])
        m = train_mlp(qt_f.transform(X_train.iloc[tr]), y_train.iloc[tr], loss_fn)
        folds.append(report(predict_proba(m, qt_f.transform(X_train.iloc[va])), y_train.iloc[va]))
    cv_rows[name] = pd.DataFrame(folds).mean()
cv_df = pd.DataFrame(cv_rows).T[["PR-AUC", "F1", "Recall", "Precision", "Brier", "p TB"]]
print(cv_df)
