import pandas as pd


root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation-Ecomerece\data\hung.csv"
root_ = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation-Ecomerece\data\one_data.csv"


df_hung = pd.read_csv(root)
df_tung = pd.read_csv(root_)

df_hung = df_hung[["productID", "name"]]
df_hung = df_hung.rename(columns={'productID': 'product_id'})

merged_df = df_tung.merge(df_hung, on='product_id', how='left')

merged_df.to_csv("data/merged_data_second.csv")