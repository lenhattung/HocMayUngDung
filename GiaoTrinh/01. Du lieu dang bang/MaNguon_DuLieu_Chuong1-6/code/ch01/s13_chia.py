# Ô lệnh 13 -- Tách đặc trưng X và nhãn y, rồi chia train/test theo tỷ lệ 80/20
from sklearn.model_selection import train_test_split

y = df["default"]
X = df.drop(columns="default").astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

print("X_train:", X_train.shape, "   X_test:", X_test.shape)
print(f"Tỷ lệ vỡ nợ -- toàn bộ: {y.mean():.4f} | train: {y_train.mean():.4f}"
      f" | test: {y_test.mean():.4f}")
