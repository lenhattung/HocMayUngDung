# Ô lệnh 14 -- Mô hình "ngây thơ": lúc nào cũng đoán lớp xuất hiện nhiều nhất
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

print("Các dự đoán:", pd.Series(pred_dummy).value_counts().to_dict())
print("Accuracy   :", round(accuracy_score(y_test, pred_dummy), 4))
