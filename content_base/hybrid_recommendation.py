import pickle

def recommendation(model_path, p_id, top_k=100):
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    recommendations = model.get(p_id, {})
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)[:top_k]
    print(recommendations.items())
    formatted_recommendations = [
        {"product_id": product_id, "similarity_score": similarity_score}
        for product_id, similarity_score in sorted_recommendations
    ]

    return formatted_recommendations