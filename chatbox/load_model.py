import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Dữ liệu của bạn
products = [
    {"product_id": 57535, "category_code": "construction tools generator", "brand": "tefal", "price": 74.65},
    {"product_id": 57534, "category_code": "apparel shoes", "brand": "ralfringer", "price": 115.06},
    {"product_id": 57533, "category_code": "apparel shoes", "brand": "ralfringer", "price": 120.72},
    {"product_id": 57532, "category_code": "apparel shoes", "brand": "ralfringer", "price": 120.72},
    {"product_id": 57531, "category_code": "computers desktop", "brand": "hp", "price": 950.35},
]

# Sử dụng mô hình SentenceTransformer để tạo embeddings cho các mô tả sản phẩm
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Mô hình nhỏ gọn và hiệu quả

# Tạo embeddings cho mỗi sản phẩm từ các thuộc tính
def create_product_embedding(product):
    description = f"{product['category_code']} {product['brand']} {product['price']}"
    return model.encode(description)

# Tạo embeddings cho tất cả các sản phẩm
embeddings = np.array([create_product_embedding(product) for product in products])

# Lưu các embeddings vào FAISS
dimension = embeddings.shape[1]  # Chiều dài của mỗi embedding
index = faiss.IndexFlatL2(dimension)  # Sử dụng khoảng cách L2 để so sánh
index.add(embeddings)

# Lưu trữ các sản phẩm theo ID
product_dict = {i: product for i, product in enumerate(products)}

def search(query, index, k=3):
    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)  # Tìm 3 sản phẩm liên quan nhất
    related_products = [product_dict[idx] for idx in indices[0]]
    return related_products

# Truy vấn tìm sản phẩm liên quan
query = "Tell me about affordable apparel shoes"
related_products = search(query, index)
for product in related_products:
    print(product)
