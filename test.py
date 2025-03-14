import pickle

# Đường dẫn đến file .pkl chứa mô hình collaborative filtering
file_path = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece\models/collaborative_09_03_25_22_07.pkl"

# Tải mô hình từ file .pkl
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Kiểm tra xem mô hình có thuộc tính trainset không
if hasattr(model, 'trainset'):
    # Lấy tất cả user IDs từ trainset
    # trainset.all_users() trả về internal IDs, cần chuyển về raw IDs
    all_users = [model.trainset.to_raw_uid(inner_id) for inner_id in model.trainset.all_users()]

    # In danh sách tất cả user IDs
    print("Danh sách tất cả user đã được huấn luyện trong mô hình:")
    for user_id in all_users:
        print(user_id)
    print(f"Tổng số user: {len(all_users)}")
else:
    print("Mô hình không chứa trainset hoặc không phải mô hình Surprise.")

# D:\Anaconda3\envs\pythonProject\python.exe D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece\qdrant_server\load_data.py
# INFO:httpx:HTTP Request: GET http://103.155.161.100:6333 "HTTP/1.1 200 OK"
# INFO:httpx:HTTP Request: GET http://103.155.161.100:6333/collections/recommendation_system/exists "HTTP/1.1 200 OK"
# INFO:root:Connect Complete
# INFO:httpx:HTTP Request: GET http://103.155.161.100:6333/collections/recommendation_system "HTTP/1.1 200 OK"
# INFO:root:Total points: 100
# INFO:httpx:HTTP Request: POST http://103.155.161.100:6333/collections/recommendation_system/points/scroll "HTTP/1.1 200 OK"
# 0     1005158
# 1     1307571
# 2     6200689
# 3     5701002
# 4     1005209
# 5    12202499
# 6     1004856
# 7    18300121
# 8     2800623
# 9     1004659
# Name: product_id, dtype: int64
# 0    574370358
# 1    558317034
# 2    517030456
# 3    572621516
# 4    579605870
# 5    515277460
# 6    543482644
# 7    514017830
# 8    572445093
# 9    555023300
# Name: user_id, dtype: int64
# INFO:root:Loaded 100 of 100 points
# INFO:root:Finished loading 100 points into DataFrame.
#
# Process finished with exit code 0
