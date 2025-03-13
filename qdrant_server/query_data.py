from qdrant_server.server import connect_qdrant

def find_vector(q_drant_end_point, q_drant_collection_name, product_id):
    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)

    result = client.scroll(
        collection_name=q_drant_collection_name,
        scroll_filter={
            "must":[
                {
                    "key": "product_id",
                    "match": {"value": product_id}
                }
            ]
        },
        limit=1,
        with_vectors=True
    )
    dictionary = {}
    if result[0]:
        for point in result[0]:
            dictionary[point.payload["product_id"]] = point.vector
            return dictionary
    else:
        print(f"No points found with p_id={product_id}")

def find_similarity(q_drant_end_point, q_drant_collection_name, vect):
    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)

    result = client.search(
        collection_name=q_drant_collection_name,
        query_vector=vect,
        limit=10,
        with_payload=True,
        with_vectors=True
    )

    if result:
        for point in result:
            return point.payload["product_id"]
    else:
        print("No similar vectors found.")

if __name__ == '__main__':
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    p_id = 1005158

    result = find_vector(q_drant_end_point, q_drant_collection_name, int(p_id))
    result = find_similarity(q_drant_end_point, q_drant_collection_name, result[p_id])
    print(result)