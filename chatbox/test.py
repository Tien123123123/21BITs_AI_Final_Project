import ollama
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Hàm trả lời câu hỏi từ người dùng
def answer_question(question):
    # Tạo context ngắn gọn từ DataFrame

    # Gửi câu hỏi và context cho OLLAMA để trả lời câu hỏi
    response = ollama.chat(
        model="llama3.2:latest",  # Chọn mô hình mà bạn muốn sử dụng
        messages=[
            {"role": "system", "content": "Hãy nhớ rằng đây là câu hỏi của user."},
            {"role": "user", "content": question}
        ]
    )
    return response['message']['content']

# Ví dụ câu hỏi từ người dùng
# while True:
# #     question = input("Enter command here (enter 0 to quit): ")
# #     if question != "0":
# #         # Trả lời câu hỏi từ người dùng
# #         answer = answer_question(question)
# #         print(f"Chatbot: {answer}")
# #     else:
# #         break

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('q')
    answer = answer_question(question)

    return jsonify(answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)