import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
import pandas as pd
from qdrant_server.server import connect_qdrant
import logging
logging.basicConfig(level=logging.INFO)

def load_to_df(client, collection_name="userbehaviors_embeddings", batch_size=1000000):
    collection_info = client.get_collection(collection_name=collection_name)
    total_points = collection_info.points_count
    logging.info(f"Total points: {total_points}")

    all_data = []
    offset = 0

    while offset < total_points:
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=min(batch_size, total_points - offset),
            offset=offset
        )
        scroll_result = scroll_result[0]

        if not scroll_result:  # Nếu không còn dữ liệu
            break

        feature_names = list(scroll_result[0].payload.keys())
        batch_data = [{
            **{feature: point.payload[feature] for feature in feature_names}
        } for point in scroll_result]

        all_data.extend(batch_data)
        offset += batch_size
        logging.info(f"Loaded {offset} of {total_points} points")
    df = pd.DataFrame(all_data)

    return df


if __name__ == '__main__':
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"

    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
    df_1 = load_to_df(client=client, collection_name=q_drant_collection_name)
    from process_data.preprocessing import preprocess_data

    df = preprocess_data(df_1)
    df = df.reset_index(drop=True)
    df.to_csv('processed_data.csv', index=False)