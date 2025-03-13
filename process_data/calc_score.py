import pandas as pd

def calc_score(df, w1, w2, w3, w4):

    # initial_weights = {"F1": 0.1, "F2": 0.25, "F3": 0.45, "F4": 0.2}
    # df["score"] = (initial_weights["F1"] * df["F1"] +
    #                initial_weights["F2"] * df["F2"] +
    #                initial_weights["F3"] * df["F3"] +
    #                initial_weights["F4"] * df["F4"])
    df["score"] = (w1 * df["F1"] +
                   w2 * df["F2"] +
                   w3 * df["F3"] +
                   w4 * df["F4"])

    df_weighted = df.groupby(["user_id", "product_id"])["score"].sum().reset_index(name="score")
    return df_weighted
