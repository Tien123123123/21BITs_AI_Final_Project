import os, sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

def recommendation(model, p_id, top_k=100):
    """
    Returns the top-k recommendations for a given product ID.

    Parameters:
        model (dict): A dictionary containing product IDs and their recommendations.
        p_id (int): The product ID for which recommendations are required.
        top_k (int): Number of top recommendations to return.

    Returns:
        dict: Formatted recommendations with the given product ID and its top-k recommendations.
    """
    # Make sure model is a dictionary or list-like structure, if necessary
    if not isinstance(model, dict):
        raise TypeError("Model must be a dictionary of product recommendations.")

    # Find the product and its recommendations
    formatted_recommendations = None
    if p_id in model:
        recs = model[p_id][:top_k]  # Get top_k recommendations
        formatted_recommendations = {
            "product_id": p_id,
            "recommendations": [{"product_id": item} for item in recs]
        }

    # If no recommendations were found for the product ID
    if not formatted_recommendations:
        return {"error": "Product ID not found in the model"}

    return formatted_recommendations


if __name__ == '__main__':
    model_path = "../models/content_base_14_03_25_21_56.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(len(model))
    # p_id = "120"
    #
    # result = recommendation(model, '1004858', top_k=10)
    # print(result)