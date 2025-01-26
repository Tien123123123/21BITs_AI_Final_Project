import os

def calculate_trending_score(df_1, df_2, file_path="data/merged_data.csv"):
    # Calculate product's interesting score
    df_trending = df_1.groupby(["product_id"])["score"].sum().reset_index(name="trending_score")
    df_trending = df_trending.rename(columns={"product_id": "ProductID"})

    # Combine interesting score into Data Frame
    df_2 = df_2.merge(df_trending, on=["ProductID"], how="left")

    # Export file
    df_2.to_csv(file_path, index=False)
    if os.path.exists(file_path):
        print(f"Data save successfully at {file_path}")
    else:
        raise ValueError("Invalid Save Path")