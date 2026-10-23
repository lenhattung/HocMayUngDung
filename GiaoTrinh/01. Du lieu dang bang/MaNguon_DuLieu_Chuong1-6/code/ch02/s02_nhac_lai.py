# Ô lệnh 2 -- Nhắc lại: XGBoost ở Chương 1 bỏ sót bao nhiêu người vỡ nợ?
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, average_precision_score)

xgb = xgb_clf().fit(X_train, y_train)
proba_base = xgb.predict_proba(X_test)[:, 1]
pred_base = (proba_base >= 0.5).astype(int)

TN, FP, FN, TP = confusion_matrix(y_test, pred_base).ravel()
print(f"Thực sự vỡ nợ: {TP + FN} người -> phát hiện được {TP}, bỏ sót {FN}")
print(f"Precision = {precision_score(y_test, pred_base):.3f}, "
      f"Recall = {recall_score(y_test, pred_base):.3f}, F1 = {f1_score(y_test, pred_base):.3f}")
