# /app/qdrant_server/load_data.py
import pandas as pd
from qdrant_client import QdrantClient
import logging

def load_to_df(client, collection_name="userbehaviors_embeddings", batch_size=10000, timeout=300):
    """Fetch data from Qdrant in batches with custom timeout."""
    # Configure client with increased timeout
    client.http.timeout = timeout  # Set timeout in seconds (5 minutes)

    collection_info = client.get_collection(collection_name=collection_name)
    total_points = collection_info.points_count
    logging.info(f"Total points: {total_points}")

    all_data = []
    offset = 0

    while offset < total_points:
        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=min(batch_size, total_points - offset),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            scroll_result = scroll_result[0]

            if not scroll_result:
                break

            feature_names = list(scroll_result[0].payload.keys())
            batch_data = [{
                **{feature: point.payload[feature] for feature in feature_names}
            } for point in scroll_result]

            all_data.extend(batch_data)
            offset += len(scroll_result)  # Use actual fetched size
            logging.info(f"Loaded {offset} of {total_points} points")
        except Exception as e:
            logging.error(f"Error fetching batch at offset {offset}: {str(e)}")
            break

    df = pd.DataFrame(all_data)
    logging.info(f"Finished loading {len(df)} points into DataFrame.")
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