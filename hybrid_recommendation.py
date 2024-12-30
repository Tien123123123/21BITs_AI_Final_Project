import pickle

import pickle
import random

def recommendation(model_path, p_id, top_k=100, random_sample_size=10):
    # Load model từ file pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    recommendations = model.get(p_id, {})
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)[:top_k]

    formatted_recommendations = [
        {"product_id": product_id, "similarity_score": similarity_score}
        for product_id, similarity_score in sorted_recommendations
    ]

    random_recommendations = random.sample(formatted_recommendations, min(random_sample_size, len(formatted_recommendations)))

    return random_recommendations


# [41967 41966 41965 ...     2     1     0]

# model_path = "models/top_k_content_base.pkl"
# p_id = 34417
# result = recommendation(model_path, p_id)
# print(result)
