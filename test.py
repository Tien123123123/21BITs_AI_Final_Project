from evaluation_pretrain.pretrain_contentbase import arg_parse_contentbase, pretrain_contentbase
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/pretrain_contentbase', methods=['POST'])
def pretrain_contentbase_api():
    try:
        data = request.get_json()

        bucket_name = data["bucket_name"]
        dataset = data["dataset"]
        k = data["k_out"]

        print(f"Obtain data successfully !")
        print(f"bucket_name: {bucket_name}")
        print(f"dataset: {dataset}")
        print(f"k: {k}")

        pretrain = pretrain_contentbase(arg_parse_contentbase(), bucket_name=bucket_name, dataset=dataset, k=k)
        return jsonify({
            "result": pretrain
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)