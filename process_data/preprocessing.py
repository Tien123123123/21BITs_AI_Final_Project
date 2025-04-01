import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from qdrant_server.server import connect_qdrant
from qdrant_server.load_data import load_to_df


def preprocess_data(df, nrows=None, is_encoded=True):
    df = df

    df = df.replace([0, '0'], np.nan).dropna()

    initial_row_count = len(df)
    df = df[~df["event_time"].str.contains(r".\d+\s", regex=True, na=False)]
    dropped_rows = initial_row_count - len(df)

    # Adjust category code
    df["category_code"] = df["category_code"].apply(lambda loc: str(loc).replace(".", " "))

    # Encode product, user, and session IDs
    df["product_id"] = df["product_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["user_session"] = df["user_session"].astype(str)
    if is_encoded:
        enc = LabelEncoder()
        df["product_id"] = enc.fit_transform(df["product_id"])
        df["user_id"] = enc.fit_transform(df["user_id"])
        df["user_session"] = enc.fit_transform(df["user_session"])

    # Calculate F1 (view), F2 (cart), F3 (purchase), F4 (time duration)
    # Separate user behaviours into behaviour cols (F1, F2, F3)
    # a. is_view, is_cart, is_purchase: number of view/cart/purchase for an item in a session
    is_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(
        name="is_view")
    is_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(
        name="is_cart")
    is_purchase = df[df["event_type"] == "purchase"].groupby(
        ["user_id", "user_session", "product_id"]).size().reset_index(name="is_purchase")

    # b. total_view, total_cart, total_purchase: number of view/cart/purchase in a session
    total_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session"]).size().reset_index(
        name="total_view")
    total_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session"]).size().reset_index(
        name="total_cart")
    total_purchase = df[df["event_type"] == "purchase"].groupby(["user_id", "user_session"]).size().reset_index(
        name="total_purchase")

    # merge all behaviour cols into original dataframe (df)
    df = (df.merge(is_view, on=["user_id", "user_session", "product_id"], how="left")
          .merge(is_cart, on=["user_id", "user_session", "product_id"], how="left")
          .merge(is_purchase, on=["user_id", "user_session", "product_id"], how="left")
          .merge(total_view, on=["user_id", "user_session"], how="left")
          .merge(total_cart, on=["user_id", "user_session"], how="left")
          .merge(total_purchase, on=["user_id", "user_session"], how="left"))

    # Drop duplicate rows
    df = df.drop_duplicates()  # Loại bỏ các dòng trùng lặp

    df.fillna(0, inplace=True)

    # Calculate duration and total duration (F4)
    # duration: time user spent for an item in a session
    # total duration: duration when user start and end a session
    df = df.sort_values(by=["user_id", "user_session", "event_time"], ascending=[True, True, True])
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["duration"] = df.groupby(["user_id", "user_session"])["event_time"].diff().fillna(pd.Timedelta(seconds=0))
    df["duration"] = df["duration"].dt.total_seconds()
    df["total_duration"] = df.groupby(["user_id", "user_session"])["duration"].transform("sum")

    # Feature calculations
    df['F1'] = np.where(df['total_view'] != 0, df['is_view'] / df['total_view'], 0)
    df['F2'] = np.where(df['total_cart'] != 0, df['is_cart'] / df['total_cart'], 0)
    df['F3'] = np.where(df['total_purchase'] != 0, df['is_purchase'] / df['total_purchase'], 0)
    df['F4'] = np.where(df['total_duration'] != 0, df['duration'] / df['total_duration'], 0)

    return df


if __name__ == '__main__':
    root = "data/one_data.csv"  # write data in here

    df_main, _ = preprocess_data(root, is_encoded=True, nrows=100)
    df_main.to_csv("processed_data.csv", index=False)  # Lưu lại dữ liệu đã xử lý vào file CSV