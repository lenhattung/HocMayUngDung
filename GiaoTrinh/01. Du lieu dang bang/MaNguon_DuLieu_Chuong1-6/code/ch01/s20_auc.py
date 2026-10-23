# Ô lệnh 20 -- ROC-AUC và PR-AUC: đánh giá mô hình trên MỌI ngưỡng cùng lúc
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

print("ROC-AUC:", round(roc_auc_score(y_test, proba_lr), 4), "  (đoán ngẫu nhiên: 0,5)")
print("PR-AUC :", round(average_precision_score(y_test, proba_lr), 4),
      f"  (đoán ngẫu nhiên: {y_test.mean():.3f})")

fpr, tpr, _ = roc_curve(y_test, proba_lr)
prec, rec, _ = precision_recall_curve(y_test, proba_lr)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(fpr, tpr)
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR (= Recall)"); ax[0].set_title("Đường ROC")
ax[1].plot(rec, prec)
ax[1].axhline(y_test.mean(), ls="--", color="gray")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Đường Precision-Recall")
plt.savefig("figures/ch01_s20_curves.pdf", bbox_inches="tight")
