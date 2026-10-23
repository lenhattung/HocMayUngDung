"""common.py -- Module dùng chung cho toàn bộ giáo trình.

Cung cấp:
  * Hàm nạp 2 bộ dữ liệu xuyên suốt: Credit Default (phân loại), California Housing (hồi quy)
  * Hàm chia train/test cố định (để mọi chương so sánh được với nhau)
  * Cấu hình chuẩn của các mô hình (XGBoost, LightGBM, hồi quy logistic)
  * Hàm tính chỉ số đánh giá, đánh giá chéo (CV)
  * Hàm vẽ hình và xuất bảng LaTeX
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn import metrics as skm

# ----------------------------------------------------------------------------
# Hằng số chung
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
RES_DIR = ROOT / "results"
for _d in (FIG_DIR, TAB_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Bảng màu (thứ tự cố định, đã kiểm tra cho người mù màu)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8984"


# ----------------------------------------------------------------------------
# 1. Nạp dữ liệu
# ----------------------------------------------------------------------------
def load_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Default of Credit Card Clients (Yeh & Lien, 2009) -- 30.000 khách hàng, 23 đặc trưng.

    Đọc file data/credit_default.xls nếu có; nếu không thì tải từ UCI (mã 350).
    Thực hiện đúng các bước làm sạch của Chương 1 (Bước 7 và Bước 9).
    """
    path = DATA_DIR / "credit_default.xls"
    if path.exists():
        df = pd.read_excel(path, header=1)
    else:  # cần kết nối Internet
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=350)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
        df.columns = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                      "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                      *[f"BILL_AMT{i}" for i in range(1, 7)],
                      *[f"PAY_AMT{i}" for i in range(1, 7)],
                      "default payment next month"]
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})
    # Gộp các mã không có trong từ điển dữ liệu vào nhóm "khác"
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # 4 = khác
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # 3 = khác
    y = df.pop("default").astype(int)
    return df.astype(float), y


def load_housing() -> tuple[pd.DataFrame, pd.Series]:
    """California Housing (Pace & Barry, 1997) -- 20.640 khu dân cư, 8 đặc trưng.

    Đọc file gốc StatLib data/cal_housing.data nếu có (biến đổi giống hệt scikit-learn);
    nếu không thì dùng sklearn.datasets.fetch_california_housing().
    """
    path = DATA_DIR / "cal_housing.data"
    cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
    if path.exists():
        raw = np.loadtxt(path, delimiter=",")[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
        target, X = raw[:, 0] / 100000.0, raw[:, 1:]
        X[:, 2] /= X[:, 5]               # AveRooms  = tổng số phòng / số hộ
        X[:, 3] /= X[:, 5]               # AveBedrms = tổng số phòng ngủ / số hộ
        X[:, 5] = X[:, 4] / X[:, 5]      # AveOccup  = dân số / số hộ
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(target, name="MedHouseVal")
    else:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        X, y = d.data[cols], d.target
    return X, y


def credit_split(test_size: float = 0.2):
    """Chia 80/20 có phân tầng (stratified), giống Bước 13 của Chương 1."""
    X, y = load_credit()
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)


def housing_split(test_size: float = 0.2):
    X, y = load_housing()
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------
# 1b. Mô hình chuẩn (cấu hình cố định, dùng lại ở mọi chương -- xem Bước 22)
# ----------------------------------------------------------------------------
def xgb_clf(**kw):
    from xgboost import XGBClassifier
    p = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
             colsample_bytree=0.8, eval_metric="logloss", n_jobs=2, random_state=RANDOM_STATE)
    p.update(kw)
    return XGBClassifier(**p)


def xgb_reg(**kw):
    from xgboost import XGBRegressor
    p = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, n_jobs=2, random_state=RANDOM_STATE)
    p.update(kw)
    return XGBRegressor(**p)


def lgbm_clf(**kw):
    from lightgbm import LGBMClassifier
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, n_jobs=2, random_state=RANDOM_STATE, verbose=-1)
    p.update(kw)
    return LGBMClassifier(**p)


def lgbm_reg(**kw):
    from lightgbm import LGBMRegressor
    p = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, n_jobs=2, random_state=RANDOM_STATE, verbose=-1)
    p.update(kw)
    return LGBMRegressor(**p)


def logreg(**kw):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))


# ----------------------------------------------------------------------------
# 2. Đánh giá
# ----------------------------------------------------------------------------
CLF_SCORING = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision",
               "F1": "f1", "Precision": "precision", "Recall": "recall",
               "BalAcc": "balanced_accuracy"}
REG_SCORING = {"RMSE": "neg_root_mean_squared_error",
               "MAE": "neg_mean_absolute_error", "R2": "r2"}


def clf_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Các chỉ số cho bài toán phân loại nhị phân mất cân bằng (Bước 17--19)."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"ROC-AUC": skm.roc_auc_score(y_true, proba),
            "PR-AUC": skm.average_precision_score(y_true, proba),
            "F1": skm.f1_score(y_true, pred),
            "Precision": skm.precision_score(y_true, pred, zero_division=0),
            "Recall": skm.recall_score(y_true, pred),
            "BalAcc": skm.balanced_accuracy_score(y_true, pred)}


def reg_metrics(y_true, y_pred) -> dict:
    return {"RMSE": skm.root_mean_squared_error(y_true, y_pred),
            "MAE": skm.mean_absolute_error(y_true, y_pred),
            "R2": skm.r2_score(y_true, y_pred)}


def cv_scores(model, X, y, task: str = "clf", n_splits: int = 5, n_jobs: int = -1) -> pd.DataFrame:
    """Đánh giá chéo k-fold (Bước 20--21). Trả về bảng: k dòng x số chỉ số."""
    if task == "clf":
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = CLF_SCORING
    else:
        cv = KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = REG_SCORING
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    out = pd.DataFrame({k: res[f"test_{k}"] for k in scoring})
    for k in ("RMSE", "MAE"):
        if k in out:
            out[k] = -out[k]
    return out


def mean_std(df: pd.DataFrame, digits: int = 3) -> pd.Series:
    """Định dạng 'trung bình ± độ lệch chuẩn' cho mỗi cột."""
    return df.apply(lambda c: f"{c.mean():.{digits}f} $\\pm$ {c.std():.{digits}f}")


# ----------------------------------------------------------------------------
# 3. Xuất kết quả
# ----------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, digits: int = 3, index: bool = True,
               bold_max: list[str] | None = None, bold_min: list[str] | None = None):
    """Ghi bảng LaTeX (booktabs) vào tables/<name>.tex để chương sách chèn trực tiếp."""
    d = df.copy()
    fmt = d.copy().astype(object)
    for c in d.columns:
        for i in d.index:
            v = d.loc[i, c]
            fmt.loc[i, c] = f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    for cols, fn in ((bold_max or [], "idxmax"), (bold_min or [], "idxmin")):
        for c in cols:
            if c in d and pd.api.types.is_numeric_dtype(d[c]):
                best = getattr(d[c], fn)()
                fmt.loc[best, c] = f"\\textbf{{{fmt.loc[best, c]}}}"
    esc = lambda s: str(s) if "$" in str(s) else str(s).replace("_", "\\_").replace("%", "\\%")
    cols = ([esc(d.index.name or "")] if index else []) + [esc(c) for c in d.columns]
    align = ("l" if index else "") + "r" * len(d.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for i in d.index:
        row = ([esc(i)] if index else []) + [fmt.loc[i, c] if "$" in fmt.loc[i, c] or "\\textbf" in fmt.loc[i, c]
                                              else esc(fmt.loc[i, c]) for c in d.columns]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(name: str, obj: dict):
    """Lưu số liệu chính ra results/<name>.json (để đối chiếu khi viết bài)."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict()
        if isinstance(o, pd.Series):
            return o.to_dict()
        return str(o)
    def clean(o):  # khóa của dict phải là chuỗi
        if isinstance(o, (pd.DataFrame, pd.Series)):
            o = json.loads(o.to_json(force_ascii=False)) if not isinstance(o.index, pd.MultiIndex) \
                else {str(k): v for k, v in o.to_dict().items()}
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = clean(obj)
    (RES_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv),
                                          encoding="utf-8")


# ----------------------------------------------------------------------------
# 4. Vẽ hình
# ----------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": "#b5b4ae", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 8,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def savefig(name: str, fig=None):
    (fig or plt.gcf()).savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig or plt.gcf())


setup_style()
