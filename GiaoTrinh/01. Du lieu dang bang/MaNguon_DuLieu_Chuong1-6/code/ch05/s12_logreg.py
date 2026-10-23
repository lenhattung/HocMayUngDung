# Ô lệnh 12 -- Với hồi quy logistic: class_weight cũng chỉ dịch chuyển sự đánh đổi
rows = []
for w_ in [1, 2, 3, round(ratio, 2), 5]:
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight={0: 1, 1: w_}))
    r = cross_validate(lr, X_train, y_train, cv=skf, scoring=scoring)
    rows.append({"w": w_, **{k: r["test_" + k].mean() for k in scoring}})
print(pd.DataFrame(rows).set_index("w"))
