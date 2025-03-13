import pandas as pd

def train_test_split(df, df_weighted, test_size=0.1):
    df_GT = df[df["event_type"] == "purchase"].drop_duplicates(subset=["user_id", "product_id"],
                                                               keep="first")  # GT contain all user has purchase item
    is_buy = df_GT[df_GT["event_type"] == "purchase"].groupby(["user_id"]).size().reset_index(name="total_buy")

    is_buy = is_buy.sort_values(by=["total_buy"], ascending=False)  # sort value
    is_buy_n = is_buy[is_buy["total_buy"] >= 3]  # Take all users who purchased more than 3 items
    is_buy_n = is_buy_n.sample(frac=test_size)  # Take N% user from the list contain user buy 3->6 items
    user_ids = is_buy_n["user_id"].unique()

    # df test contain N% users purchased more than 3 items
    # df weighted contain the rest of data except users in df test

    df_test = df_weighted[df_weighted["user_id"].isin(user_ids)]
    df_weighted = df_weighted[~df_weighted["user_id"].isin(user_ids)]

    return df_test, df_weighted, df_GT