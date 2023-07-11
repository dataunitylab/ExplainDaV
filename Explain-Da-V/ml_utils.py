import io
import re
import sys
import time

import featuretools as ft
import numpy as np
import pandas as pd
from config import *
from sklearn import svm, tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.tree import _tree

# lasso, ridge, regular, LassoLarsCV, ElasticNetCV
first_regressor = LassoLarsCV
second_regressor = RidgeCV


def powerset(s):
    out = list()
    x = len(s)
    for i in range(1 << x):
        out.append([s[j] for j in range(x) if (i & (1 << j))])
    return out


def get_formula(model, feature_names=None):
    formulas = []
    if type(model.intercept_) == np.float64:
        if abs(model.intercept_) < 0.00000000001:
            s = "y = "
        else:
            s = "y = {0:.3f}".format(model.intercept_)
        if feature_names:
            for i, c in zip(feature_names, model.coef_):
                if np.array(c) == np.array(0.0):
                    continue
                if abs(c) < 0.00000000001:
                    continue
                if abs(c) < 0.0001:
                    s += " + \u03B5 * {1}".format(c, i)
                elif c == 1.0:
                    s += " + {1}".format(c, i)
                else:
                    s += " + {0:.3f} * {1}".format(c, i)
        else:
            for i, c in enumerate(model.coef_):
                if np.array(c) == np.array(0.0):
                    continue
                s += " + {0:.3f} * x{1}".format(c, i)
        formulas.append(s)
    else:
        for intercept, coef in zip(model.intercept_, model.coef_):
            s = "y = {0:.3f}".format(intercept)
            for i, c in enumerate(coef):
                s += " + {0:.3f} * x{1}".format(c, i)
            formulas.append(s)
    return formulas[0]


def get_rules(model, feature_names, class_names):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i is not None else "undefined!" for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] is not None:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("transform({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else ({} > {}): ".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}new value: {}".format(indent, np.argmax(tree_.value[node])))

    recurse(0, 1)


def get_formula_tree_model(model, X, y):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    tree_to_code(model, X.columns.tolist())
    tree_formulation = new_stdout.getvalue()
    sys.stdout = old_stdout
    return tree_formulation


def fix_labels(y):
    numeric_y_domain = y.unique()
    exchange_dict = dict(zip(numeric_y_domain, range(len(numeric_y_domain))))
    numeric_y = y.replace(exchange_dict)
    return numeric_y


def fix_domains(X):
    fixed_X = X.copy()
    for x_i in X.columns:
        if fixed_X[x_i].isnull().values.any():
            fixed_X[x_i] = fixed_X[x_i].fillna(-1.0)
        if pd.api.types.is_numeric_dtype(X[x_i]):
            fixed_X[x_i] = X[x_i]
        else:
            one_hot_rep = pd.get_dummies(X[x_i], prefix=str(x_i)).astype(int)
            fixed_X[one_hot_rep.columns.tolist()] = one_hot_rep
            fixed_X = fixed_X.drop(columns=[x_i])
    return fixed_X


def get_explainabilty_classification(clf):
    eval_explainabilty_size = clf.tree_.node_count
    features = list(set([f for f in clf.tree_.feature if f != -2]))
    eval_explainabilty_repeated_terms = len(features)
    eval_explainabilty_cognitive_chunks = eval_explainabilty_size - clf.get_n_leaves()
    return (
        eval_explainabilty_size,
        eval_explainabilty_repeated_terms,
        eval_explainabilty_cognitive_chunks,
    )


def learn_data_transformation_BINARY_CLASSIFICATION(
    X_train, y_train, X_test, y_test, applied_over_rows=False
):
    # TODO: try XGBOOST?? https://towardsdatascience.com/xgboost-deployment-made-easy-6e11f4b3f817
    start = time.time()
    if X_train.empty or len(y_train) == 0:
        return {}
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    formula = get_formula_tree_model(clf, X_train, y_train)
    eval_explainabilty = get_explainabilty_classification(clf)
    eval_validation = clf.score(X_train, y_train)
    eval_generalization = clf.score(X_test, y_test)
    if applied_over_rows:
        validation_labels = dict(zip(X_train.index, clf.predict(X_train)))
        generalization_labels = dict(zip(X_test.index, clf.predict(X_test)))
        return formula, validation_labels, generalization_labels
    run_time = time.time() - start
    return {
        formula: [eval_validation, eval_generalization, *eval_explainabilty, run_time]
    }


def learn_data_transformation_MULTICLASS_CLASSIFICATION(
    X_train, y_train, X_test, y_test, include_aggregated=False, non_numeric=None
):
    # TODO: try XGBOOST??
    start = time.time()
    models = {}
    clf = tree.DecisionTreeClassifier()
    if y_train.dtype != y_test.dtype:
        return models
    if X_train.empty or len(y_train) == 0:
        return {}

    clf.fit(X_train, y_train)
    eval_explainabilty = get_explainabilty_classification(clf)
    formula = get_formula_tree_model(clf, X_train, y_train)
    # eval_simplicity = 1.0
    eval_validation = clf.score(X_train, y_train)
    try:
        eval_generalization = clf.score(X_test, y_test)
    except:
        eval_generalization = 0.0
    run_time = time.time() - start
    models[formula] = [
        eval_validation,
        eval_generalization,
        *eval_explainabilty,
        run_time,
    ]
    return models


def get_explainabilty_regression(regr, formula):
    eval_explainabilty_size = float(
        len(
            [
                c
                for c in regr.coef_
                if float(abs(c)) > 0.00000000001 and np.array(c) != np.array(0.0)
            ]
        )
    )

    if regr.intercept_ != 0.0:
        eval_explainabilty_size += 1
    vars = set(re.findall("x(\d+)", formula))
    eval_explainabilty_repeated_terms = float(len(vars))
    eval_explainabilty_cognitive_chunks = float(bool(eval_explainabilty_size))
    if len(vars) == 0:
        power_max = 0
    else:
        power_max = 1
    for power in range(2, 5):
        if "^{}".format(power) in formula:
            power_max = power

    eval_explainabilty_cognitive_chunks += float(power_max)
    eval_explainabilty_cognitive_chunks += float("log" in formula)
    eval_explainabilty_cognitive_chunks += float("sqrt" in formula)
    eval_explainabilty_cognitive_chunks += float("1/" in formula)
    eval_explainabilty_cognitive_chunks += float("exp" in formula)
    eval_explainabilty_cognitive_chunks += bool(
        len(re.findall("x(\d+) x(\d+)", formula))
    )
    eval_explainabilty_cognitive_chunks += bool(
        len(re.findall("x(\d+)/x(\d+)", formula))
    )

    return (
        eval_explainabilty_size,
        eval_explainabilty_repeated_terms,
        eval_explainabilty_cognitive_chunks,
    )


def compute_score(X, y, regr):
    eval_validation = 0
    try:
        eval_validation = regr.score(X, y)
    except Exception as err:
        if "Input contains NaN, infinity or a value too large" in str(err):
            return 0
    if eval_validation < 0:
        eval_validation = 0
    return eval_validation


def learn_data_transformation_REGRESSION_step(
    X_train, y_train, X_test, y_test, feature_names=None, regr_type=RidgeCV
):
    start = time.time()
    regr = regr_type()

    try:
        if np.any(np.isnan(X_train)):
            X_train = X_train.fillna(0.0)
        if np.any(np.isnan(y_train)):
            y_train = y_train.fillna(0.0)
        if np.any(np.isnan(X_test)):
            X_test = X_test.fillna(0.0)
        if np.any(np.isnan(y_test)):
            y_test = y_test.fillna(0.0)
    except:
        X_train = X_train
        y_train = y_train
        X_test = X_test
        y_test = y_test
    try:
        y_train.astype(np.float64)
        y_test.astype(np.float64)
    except:
        return {}
    regr.fit(X_train, y_train)
    formula = get_formula(regr, feature_names)
    eval_explainabilty = get_explainabilty_regression(regr, formula)
    eval_validation = compute_score(X_train, y_train, regr)
    eval_generalization = compute_score(X_test, y_test, regr)
    run_time = time.time() - start

    return {
        formula: [eval_validation, eval_generalization, *eval_explainabilty, run_time]
    }


def learn_data_transformation_REGRESSION(
    X_train,
    y_train,
    X_test,
    y_test,
    binary_features=None,
    include_grouping_aggregated=False,
    non_numeric=None,
):
    models = {}
    if X_train.empty or len(y_train) == 0:
        return {}

    if include_grouping_aggregated:
        for a in non_numeric:
            X_train_a = extend_features_with_local_aggregations(X_train.copy(), a)
            X_test_a = extend_features_with_local_aggregations(X_test.copy(), a)
            feature_names = list(X_train_a.columns)
            try:
                models.update(
                    learn_data_transformation_REGRESSION_step(
                        X_train_a,
                        y_train,
                        X_test_a,
                        y_test,
                        feature_names,
                        first_regressor,
                    )
                )
            except:
                models.update(
                    learn_data_transformation_REGRESSION_step(
                        X_train_a,
                        y_train,
                        X_test_a,
                        y_test,
                        feature_names,
                        second_regressor,
                    )
                )
        return models
    try:
        sol = learn_data_transformation_REGRESSION_step(
            X_train, y_train, X_test, y_test, list(X_train.columns), first_regressor
        )
        sol = {"(no extensions)" + k: sol[k] for k in sol}
        models.update(sol)
    except:
        sol = learn_data_transformation_REGRESSION_step(
            X_train, y_train, X_test, y_test, list(X_train.columns), second_regressor
        )
        sol = {"(no extensions)" + k: sol[k] for k in sol}
        models.update(sol)
    if binary_features:
        non_binary_features = [a for a in X_train.columns if a not in binary_features]
        if len(non_binary_features) == 0:
            return models
        X_train = X_train[non_binary_features]
    extend_features_poly_2 = lambda X: extend_features_poly(X, 2)
    extend_features_poly_3 = lambda X: extend_features_poly(X, 3)
    all_possible_extensions = [
        extend_features_poly_2,
        extend_features_poly_3,
        extend_features_div,
        extend_features_additional,
        extend_features_global_aggregations,
    ]
    for extensions in powerset(all_possible_extensions):
        if not len(extensions):
            continue
        elif len(extensions) > 1:
            continue
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        for extend in extensions:
            X_train_new = extend(X_train_new)
            X_test_new = extend(X_test_new)
            names = list(X_train_new.columns)
        try:
            sol = learn_data_transformation_REGRESSION_step(
                X_train_new, y_train, X_test_new, y_test, names, first_regressor
            )
            models.update(sol)
        except:
            sol = learn_data_transformation_REGRESSION_step(
                X_train_new, y_train, X_test_new, y_test, names, second_regressor
            )
            models.update(sol)
    return models


def extend_features_poly(X, poly_size=2):
    poly = PolynomialFeatures(poly_size, include_bias=False)
    extended_X = poly.fit_transform(X)
    names = poly.get_feature_names_out()
    extended_X = np.nan_to_num(extended_X)
    extended_X = pd.DataFrame(extended_X, columns=names)
    extended_X = extended_X.mask(extended_X > new_inf, new_inf)
    extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)

    return extended_X


def extend_features_div(X):
    extended_X = X
    names = ["x" + str(i) for i, _ in enumerate(X.columns)]
    for i, col in enumerate(X.columns):
        if X[col].dtype == "category":
            continue
        for j, col_j in enumerate(X.columns):
            if j == i:
                continue
            if X[col_j].dtype == "category":
                continue
            else:
                extended_X = np.column_stack(
                    (
                        extended_X,
                        np.divide(
                            X[col],
                            X[col_j],
                            out=np.zeros_like(X[col]),
                            where=X[col_j] != 0,
                        ),
                    )
                )
                names.append("x" + str(i) + "/x" + str(j))
    extended_X = np.nan_to_num(extended_X)
    extended_X = pd.DataFrame(extended_X, columns=names)
    extended_X = extended_X.mask(extended_X > new_inf, new_inf)
    extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)

    return extended_X


def extend_features_additional(X):
    extended_X = X
    names = ["x" + str(i) for i, _ in enumerate(X.columns)]
    for i, col in enumerate(X.columns):
        if X[col].dtype != float:
            if X[col].dtype == int:
                X[col] = X[col].astype(float)
            else:
                continue
        extended_X = np.column_stack((extended_X, np.log(X[col])))
        names.append("log(x" + str(i) + ")")
        extended_X = np.column_stack((extended_X, np.sqrt(X[col])))
        names.append("sqrt(x" + str(i) + ")")
        extended_X = np.column_stack((extended_X, np.reciprocal(X[col])))
        names.append("1/x" + str(i))
        extended_X = np.column_stack((extended_X, np.exp(X[col])))
        names.append("exp(x" + str(i) + ")")
    extended_X = np.nan_to_num(extended_X)
    extended_X = pd.DataFrame(extended_X, columns=names)
    extended_X = extended_X.mask(extended_X > new_inf, new_inf)
    extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)
    return extended_X


def extend_features_global_aggregations(X):
    extended_X = X
    extended_X["all"] = 1
    es = ft.EntitySet(id="T")
    es.add_dataframe(dataframe_name="T", dataframe=X, make_index=True, index="index")
    es.normalize_dataframe(new_dataframe_name=" ", base_dataframe_name="T", index="all")
    extended_X = ft.dfs(target_dataframe_name="T", entityset=es)[0].drop(
        ["all"], axis=1
    )

    try:
        extended_X = extended_X.mask(extended_X > new_inf, new_inf)
        extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)
    except:
        pass
    return extended_X


def extend_features_with_local_aggregations(X, non_numeric):
    extended_X = X
    es = ft.EntitySet(id="T")
    es.add_dataframe(
        dataframe_name="T", dataframe=extended_X, make_index=True, index="index"
    )
    es.normalize_dataframe(
        new_dataframe_name=non_numeric, base_dataframe_name="T", index=non_numeric
    )
    extended_X = ft.dfs(target_dataframe_name="T", entityset=es)[0].select_dtypes(
        ["number"]
    )

    try:
        extended_X = extended_X.fillna(0.0)
        extended_X = extended_X.mask(extended_X > new_inf, new_inf)
        extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)
    except:
        extended_X = extended_X
    return extended_X


def explore_group_bys_single(X, group_by_candidate_name):
    extended_X = X
    es = ft.EntitySet(id="T")
    es.add_dataframe(
        dataframe_name="T_" + group_by_candidate_name,
        dataframe=extended_X,
        make_index=True,
        index="index",
    )

    es.normalize_dataframe(
        new_dataframe_name=group_by_candidate_name,
        base_dataframe_name="T_" + group_by_candidate_name,
        index=group_by_candidate_name,
    )
    extended_X = (
        ft.dfs(target_dataframe_name=group_by_candidate_name, entityset=es)[0]
        .reset_index()
        .sort_values(group_by_candidate_name)
    )
    extended_X = extended_X.drop([group_by_candidate_name], axis=1).select_dtypes(
        ["number"]
    )
    extended_X = extended_X.fillna(0.0)
    extended_X = extended_X.mask(extended_X > new_inf, new_inf)
    extended_X = extended_X.mask(extended_X < -new_inf, -new_inf)

    return extended_X


def explore_group_bys_multi(X, group_by_candidate_names):
    concat_group_by_candidate_names = "+".join(group_by_candidate_names)
    extended_X = X
    extended_X[concat_group_by_candidate_names] = extended_X[
        group_by_candidate_names[0]
    ].astype(str)
    for candidate_name in group_by_candidate_names[1:]:
        extended_X[concat_group_by_candidate_names] += "+" + extended_X[
            candidate_name
        ].astype(str)
    es = ft.EntitySet(id="T")
    es.add_dataframe(
        dataframe_name="T_" + concat_group_by_candidate_names,
        dataframe=extended_X,
        make_index=True,
        index="index",
    )
    es.normalize_dataframe(
        new_dataframe_name=concat_group_by_candidate_names,
        base_dataframe_name="T_" + concat_group_by_candidate_names,
        index=concat_group_by_candidate_names,
    )

    extended_X = ft.dfs(
        target_dataframe_name=concat_group_by_candidate_names, entityset=es
    )[0].reset_index()
    for i, cand in enumerate(group_by_candidate_names):
        extended_X[cand] = (
            extended_X.reset_index()[concat_group_by_candidate_names]
            .str.split("+")
            .str[i]
        )
    extended_X = extended_X.sort_values(group_by_candidate_names)
    extended_X = extended_X.select_dtypes(["number"])
    extended_X = extended_X.fillna(0.0)
    extended_X.loc[extended_X > new_inf, :] = new_inf
    extended_X.loc[extended_X < -new_inf, :] = -new_inf
    return extended_X


def learn_data_transformation_REGRESSION_projected_table(
    X_train, y_train, X_test, y_test, group_by_candidate
):
    models = {}
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    if len(group_by_candidate) == 1:
        group_by_candidate_name = X_train_new.columns.tolist()[group_by_candidate[0]]
        X_train_new = explore_group_bys_single(X_train_new, group_by_candidate_name)
        X_test_new = explore_group_bys_single(X_test_new, group_by_candidate_name)
    else:
        group_by_candidate_names = [
            X_train_new.columns.tolist()[i] for i in group_by_candidate
        ]
        X_train_new = explore_group_bys_multi(X_train_new, group_by_candidate_names)
        X_test_new = explore_group_bys_multi(X_test_new, group_by_candidate_names)
    feature_names = list(X_train_new.columns)
    if len(X_train_new) != len(y_train):
        return {}
    if feature_names != list(X_test_new.columns):
        for delta_col in feature_names:
            if delta_col in list(X_test_new.columns):
                continue
            X_test_new[delta_col] = 0.0
    if len(X_train_new) < 100:
        merged = (
            X_train_new.reset_index()
            .merge(y_train, left_index=True, right_index=True)
            .sample(n=10, random_state=1, replace=True)
        )
        X_train_new = merged[X_train_new.columns]
        y_train = merged[y_train.name]
    try:
        models.update(
            learn_data_transformation_REGRESSION_step(
                X_train_new, y_train, X_test_new, y_test, feature_names, first_regressor
            )
        )
    except:
        models.update(
            learn_data_transformation_REGRESSION_step(
                X_train_new,
                y_train,
                X_test_new,
                y_test,
                feature_names,
                second_regressor,
            )
        )
    return models
