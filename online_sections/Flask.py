import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from flask import Flask, request, jsonify
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase, arg_parse_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative, arg_parse_collaborative

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

        pretrain = pretrain_contentbase(arg_parse_contentbase, bucket_name=bucket_name, dataset=dataset, k=k)
        return jsonify({
            "result": pretrain
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/pretrain_collaborative', methods=['POST'])
def pretrain_collaborative_api():
    try:
        data = request.get_json()

        bucket_name = data["bucket_name"]
        dataset = data["dataset"]

        print(f"Obtain data successfully !")
        print(f"bucket_name: {bucket_name}")
        print(f"dataset: {dataset}")

        pretrain = pretrain_collaborative(arg_parse_collaborative, bucket_name=bucket_name, dataset=dataset)
        return jsonify({
            "result": pretrain
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)


