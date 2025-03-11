import pandas as pd

# Đường dẫn đến file CSV
path = "D:/Pycharm/Projects/pythonProject/AI/ML/Projects/Recommendation_Ecomerece/data/processed_data.csv"

# Đọc file CSV
df = pd.read_csv(path)

# Loại bỏ cột Unnamed: 0 nếu nó tồn tại
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# In DataFrame để kiểm tra
# df=df[:10]
# df.to_csv("10_items.csv")

# Hàm tạo context từ DataFrame
def create_context_from_df(df):
    context = ""
    for index, row in df.iterrows():
        context += f"Sản phẩm ID: {row['product_id']}, Loại: {row['category_code']}, Thương hiệu: {row['brand']}, Giá: {row['price']}\n"
    return context

# Tạo context và in ra
context = create_context_from_df(df)
print(context)
print(len(context))
print(len(df))