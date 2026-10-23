# Ô lệnh 15 -- Hồi quy logistic, có chuẩn hóa dữ liệu trước khi học
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred_lr), 4))
print("Số khách hàng bị dự đoán là vỡ nợ:", pred_lr.sum(), "/", len(pred_lr))
print("Số khách hàng thực sự vỡ nợ      :", y_test.sum())
