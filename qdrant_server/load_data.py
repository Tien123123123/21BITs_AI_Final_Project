import pandas as pd
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_server.server import connect_qdrant

def load_to_df(client: QdrantClient, collection_name="userbehaviors_embeddings", batch_size=1000000, timeout=300):
    client.http.timeout = timeout  # Set custom timeout in seconds
    all_data = []
    scroll_offset = None  # Start with None
    limit = 0
    seen_point_ids = set()  # Track point IDs to avoid duplicates

    try:
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.points_count
        logging.info(f"Total points in collection '{collection_name}': {total_points}")
    except Exception as e:
        logging.error(f"Failed to retrieve collection info: {str(e)}")
        return pd.DataFrame()

    while limit < total_points:
        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=min(batch_size, total_points - limit),
                offset=scroll_offset,  # Use cursor-based scrolling
                with_payload=True,
                with_vectors=False
            )

            points, next_offset = scroll_result

            if not points:
                break  # No more data

            # Process the batch, skipping duplicates
            batch_data = []
            for point in points:
                if point.id in seen_point_ids:
                    logging.warning(f"Duplicate point ID found: {point.id}, skipping")
                    continue
                seen_point_ids.add(point.id)
                feature_names = list(point.payload.keys())
                batch_data.append({
                    **{feature: point.payload.get(feature, None) for feature in feature_names}
                })

            all_data.extend(batch_data)
            limit += len(points)  # Correctly increment by the size of the current batch
            logging.info(f"Loaded {len(all_data)} / {total_points} points...")

            # Update the scroll offset
            scroll_offset = next_offset
            if scroll_offset is None:
                break  # Done scrolling
        except Exception as e:
            logging.error(f"Error fetching data from Qdrant: {str(e)}")
            break

    df = pd.DataFrame(all_data)
    logging.info(f"Finished loading {len(df)} total points into DataFrame.")
    unique_users1 = df['user_id'].nunique()
    logging.info(f"unique user: {unique_users1}")
    unique_products1 = df['product_id'].nunique()
    logging.info(f"unique product: {unique_products1}")
    return df

if __name__ == '__main__':
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"

    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
    df = load_to_df(client=client, collection_name=q_drant_collection_name)

    u_id = df["user_id"].unique()
    p_id = df["product_id"].unique()
    print(f"Before drop is: {len(df)}")
    print(f"Before user is: {len(u_id)}")
    print(f"Before product is: {len(p_id)}")

    df = df.drop_duplicates(subset=["user_id", "product_id"], keep="first")

    print(f"After drop is: {len(df)}")
    u_id = df["user_id"].unique()
    p_id = df["product_id"].unique()
    print(f"After user is: {len(u_id)}")
    print(f"After product is: {len(p_id)}")