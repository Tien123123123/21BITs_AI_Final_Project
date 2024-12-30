from flask import Flask, request, jsonify
from hybrid_recommendation import recommendation

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        product_id = data.get('product_id')
        model_path = "models/top_k_content_base.pkl"


        recommendations = recommendation(model_path, product_id)
        return jsonify({
            'recommendations': recommendations
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


