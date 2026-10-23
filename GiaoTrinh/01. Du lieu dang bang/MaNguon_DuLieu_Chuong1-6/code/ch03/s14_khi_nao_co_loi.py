# Ô lệnh 14 -- Khi nào Borderline-SMOTE thắng SMOTE? Thử với các mức chồng lấn khác nhau
rows = []
for sep in [0.5, 1.0, 1.5, 2.0]:                 # class_sep càng lớn, hai lớp càng tách biệt
    Xs, ys = make_classification(n_samples=4000, n_features=10, n_informative=5,
                                 weights=[0.9, 0.1], class_sep=sep, flip_y=0, random_state=0)
    t, _ = minority_types(StandardScaler().fit_transform(Xs), ys)
    row = {"class_sep": sep, "% DANGER": np.mean(t == "DANGER") * 100}
    for sname in ["SMOTE", "Borderline-1"]:
        r = cross_validate(make(xgb_clf(), samplers[sname]), Xs, ys, cv=skf, scoring=scoring)
        row[f"F1 {sname}"] = r["test_F1"].mean()
        row[f"PR-AUC {sname}"] = r["test_PR-AUC"].mean()
    rows.append(row)
print(pd.DataFrame(rows).set_index("class_sep").round(3).to_string())
