# Ô lệnh 10 -- Ảnh hưởng của số láng giềng K (XGBoost + ADASYN)
rows = []
for K_ in [3, 5, 10, 20]:
    r_ = cross_validate(make(xgb_clf(), ADASYN(n_neighbors=K_, random_state=SEED)),
                        X_train, y_train, cv=skf, scoring=scoring)
    rows.append({"K": K_, **{k: r_["test_" + k].mean() for k in scoring}})
print(pd.DataFrame(rows).set_index("K"))
