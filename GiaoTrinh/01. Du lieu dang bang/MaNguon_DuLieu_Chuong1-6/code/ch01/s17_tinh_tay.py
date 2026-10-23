# Ô lệnh 17 -- Tính Precision, Recall, F1 bằng tay rồi đối chiếu với scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print(f"Tính tay    : Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
print(f"scikit-learn: Precision = {precision_score(y_test, pred_lr):.4f}, "
      f"Recall = {recall_score(y_test, pred_lr):.4f}, F1 = {f1_score(y_test, pred_lr):.4f}")
