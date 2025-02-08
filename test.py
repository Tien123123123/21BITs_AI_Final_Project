from evaluation_pretrain.pretrain_collaborative import arg_parse_collaborative, pretrain_collaborative
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/pretrain_collaborative', methods=['POST'])
def pretrain_collaborative_api():
    try:
        data = request.get_json()

        bucket_name = data["bucket_name"]
        dataset = data["dataset"]

        print(f"Obtain data successfully !")
        print(f"bucket_name: {bucket_name}")
        print(f"dataset: {dataset}")

        pretrain = pretrain_collaborative(arg_parse_collaborative(), bucket_name=bucket_name, dataset=dataset)
        return jsonify({
            "result": pretrain
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)