import random

import numpy as np
import pandas as pd
from config import *


def load_problem_sets_generate_table_pairs(problem_sets_file):
    prefix = problem_sets_file.rsplit("/", 2)[0]
    if dataset_name == "Auto-pipeline":
        prefix = problem_sets_file.rsplit("/", 1)[0]
    # print(prefix)
    problem_sets_meta = pd.read_csv(problem_sets_file).to_dict("records")
    problem_sets = []
    for problem_set in problem_sets_meta:
        set_to_add = {}
        for data_type in [table for table in problem_set if table != "Setup"]:
            set_to_add[data_type] = pd.read_csv(
                prefix + "/" + str(problem_set[data_type])
            )
        set_to_add["Setup"] = problem_set["Setup"]
        problem_sets.append(set_to_add)
    return problem_sets


def table_pair_generation_wine_quality(LH_side_file, RH_side_file=None):
    df = pd.read_csv(LH_side_file)
    df = df[["country", "title", "price", "variety", "points"]]
    df_prime = df.copy()

    df["country_exact_copy"] = df["country"]
    df["country_with_noise"] = df["country"]
    noise_size = int(noise_rate_for_duplicate_column * len(df))
    sampled_data = (
        df["country_with_noise"].sample(n=noise_size, random_state=1).tolist()
    )
    random_ixs = random.sample(range(len(df)), noise_size)
    df["country_with_noise"].iloc[random_ixs] = sampled_data

    print(len(df_prime))
    # Row operations:
    df_prime = df_prime.drop_duplicates(["title"])
    print(len(df_prime))
    df_prime = df_prime.dropna()
    print(len(df_prime))
    df_prime = df_prime[df_prime["price"] <= 100]
    print(len(df_prime))

    df_prime = df_prime.reset_index()
    df_prime["orig_index"] = df_prime["index"]
    df_prime = df_prime.drop(["index"], axis=1)

    bootstrapped_data = df_prime.sample(n=bootstrap_size, random_state=1)
    bootstrapped_data["orig_index"] = list(range(len(df), len(df) + bootstrap_size))
    bootstrapped_data.index = list(range(len(df), len(df) + bootstrap_size))
    df_prime = df_prime.append(bootstrapped_data)

    # TODO: CAN WE ALSO HANDLE: df_prime.loc[:, 'is_white_wine'] = df_prime['title'].str.lower().str.contains('white').astype(int)

    # String extraction:
    df_prime.loc[:, "year"] = df_prime["title"].str.extract(
        "([1-3][0-9]{3})", expand=False
    )

    # Numeric Operations:
    df_prime.loc[:, "points_power_2_plus_price"] = (
        df_prime["points"] * df_prime["points"]
    ) + df_prime["price"]
    df_prime.loc[:, "points_divide_by_price"] = df_prime["points"] / df_prime["price"]
    df_prime.loc[:, "log_price"] = np.log(df_prime["price"])
    df_prime.loc[:, "1_div_price"] = np.reciprocal(df_prime["price"])
    df_prime.loc[:, "price_norm_sum"] = df_prime["price"] / df_prime["price"].sum()
    df_prime.loc[:, "price_norm_minmax"] = (
        df_prime["price"] - df_prime["price"].min()
    ) / (df_prime["price"].max() - df_prime["price"].min())
    df_prime["is_top_quality"] = 0
    df_prime.loc[df_prime["points"] > 90, "is_top_quality"] = 1
    df_prime.loc[:, "price_plus_points"] = df_prime["price"] + df_prime["points"]
    df_prime.loc[:, "double_points_plus_10"] = 2 * df_prime["points"] + 10
    # df_prime.loc[:, 'points_on_1_to_5_scale'] = pd.qcut(df_prime['points'], 5, labels=False)
    df_prime.loc[:, "points_on_1_to_3_scale"] = pd.qcut(
        df_prime["points"], 3, labels=False
    )
    df_prime.loc[:, "random_score"] = np.random.randint(0, 100, size=(len(df_prime), 1))

    new_columns_order = [col for col in df_prime.columns if col != "orig_index"] + [
        "orig_index",
    ]
    df_prime = df_prime[new_columns_order]
    return df, df_prime


def table_pair_generation_wine_quality_full(LH_side_file, RH_side_file=None):
    df = pd.read_csv(LH_side_file)
    df = df[["country", "title", "price", "variety", "points", "region_1", "region_2"]]
    # TRANSPOSE
    df = df.fillna("")
    df = df.iloc[:number_of_rows_for_foofah, :]
    df_prime = df.T

    df_prime = df_prime.reset_index()
    df_prime["orig_index"] = df_prime["index"]
    df_prime = df_prime.drop(["index"], axis=1)
    return df, df_prime
