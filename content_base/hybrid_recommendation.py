import pickle

def recommendation(model_path, p_id, top_k=100):
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    for product_id, recs in list(model.items()):
        if product_id == p_id:
            formatted_recommendations = {
                    "product_id": product_id,
                    "recommendations": [{"product_id": item} for item in recs]
            }

    return formatted_recommendations

result = recommendation("D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece\content_base/models/content_base.pkl", 100)

print(f"this is a {result}")