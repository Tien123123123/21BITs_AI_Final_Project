
import pandas as pd
import pickle
import random
import logging
# Global variable to hold the current cold start cluster.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
CURRENT_COLD_CLUSTER = None


def recommend_products(cf_model,cold_start_clusters ,df, target_user_id, target_product_id, top_n=10):
    df["event_time"] = pd.to_datetime(df["event_time"])


    product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])

    user_exists = target_user_id in cf_model.trainset._raw2inner_id_users
    target_product_id= str(target_product_id)
    if user_exists:
        logging.info("User exists. CF session based recommendation.")
        target_max_date = df.loc[df["product_id"] == target_product_id, "event_time"].max()
        if pd.isna(target_max_date):
            logging.info(f"Warning: target_product_id={target_product_id} never appeared in the data.")
            return pd.DataFrame()

        cutoff_date = target_max_date - pd.DateOffset(months=3)
        df_recent = df[(df["event_time"] >= cutoff_date) & (df["event_time"] <= target_max_date)]
        target_sessions = df_recent.loc[df_recent["product_id"] == target_product_id, "user_session"].unique()
        related_sessions = df_recent[df_recent["user_session"].isin(target_sessions)]

        candidate_products = related_sessions["product_id"].unique().tolist()
        # Remove the target product from candidate list.
        if target_product_id in candidate_products:
            candidate_products.remove(target_product_id)

        predicted_scores = []
        for p in candidate_products:
            try:
                score = cf_model.predict(target_user_id, p).est
                predicted_scores.append((p, score))
            except Exception:
                continue

        appearance_counts = related_sessions["product_id"].value_counts()
        predicted_df = pd.DataFrame(predicted_scores, columns=["product_id", "predicted_score"])
        predicted_df = predicted_df.merge(product_details, on="product_id", how="left")
        predicted_df["appearance_count"] = predicted_df["product_id"].map(appearance_counts)

        high_freq = predicted_df[predicted_df["appearance_count"] >= 3]
        high_freq = high_freq.sort_values(by=["appearance_count", "predicted_score"], ascending=[False, False])
        top_high = high_freq.head(3)
        remaining = predicted_df[~predicted_df["product_id"].isin(top_high["product_id"])]
        remaining = remaining.sort_values(by="predicted_score", ascending=False)

        recommendations = pd.concat([top_high, remaining]).head(top_n)
        # Filter out the target product if it slipped through.
        recommendations = recommendations[recommendations["product_id"] != target_product_id]

        if len(recommendations) < top_n:
            additional_needed = top_n - len(recommendations)
            leftover = [pid for pid in df["product_id"].unique()
                        if pid not in recommendations["product_id"].values and pid != target_product_id]
            additional_scores = []
            for pid in leftover:
                try:
                    score = cf_model.predict(target_user_id, pid).est
                    additional_scores.append((pid, score))
                except Exception:
                    continue
            add_df = pd.DataFrame(additional_scores, columns=["product_id", "predicted_score"])
            add_df = add_df.merge(product_details, on="product_id", how="left")
            add_df = add_df.sort_values(by="predicted_score", ascending=False)
            add_top = add_df.head(additional_needed)
            recommendations = pd.concat([recommendations, add_top])
            recommendations = recommendations[recommendations["product_id"] != target_product_id]

        return recommendations.head(top_n)

    else:
        # Use cold start strategy for unknown users.
        logging.info("User not found in SVD model. Using Cold-Start Recommendation.")

        type_target_product_id = type(target_product_id)
        logging.info(f'type of target_product_id:{type_target_product_id }')
        target_product_id = str(target_product_id)
        logging.info(f'target_product_id:{target_product_id}')


        global CURRENT_COLD_CLUSTER
        if (CURRENT_COLD_CLUSTER is None or
                (target_product_id not in CURRENT_COLD_CLUSTER.get("top", []) and
                 target_product_id not in CURRENT_COLD_CLUSTER.get("others", []))):
            if target_product_id in cold_start_clusters:
                CURRENT_COLD_CLUSTER = cold_start_clusters[target_product_id]
                logging.info(f'CURRENT_COLD_CLUSTER:{CURRENT_COLD_CLUSTER}')
            else:
                for clust in cold_start_clusters.values():
                    if target_product_id in clust.get("top", []) or target_product_id in clust.get("others", []):
                        CURRENT_COLD_CLUSTER = clust
                        break

        cluster = CURRENT_COLD_CLUSTER
        top_candidates = cluster.get("top", [])
        others_candidates = cluster.get("others", [])

        selected_top = [pid for pid in top_candidates[:8] if pid != target_product_id]
        filtered_others = [pid for pid in others_candidates if pid != target_product_id]
        selected_others = random.sample(filtered_others, 2) if len(filtered_others) >= 2 else filtered_others
        rec_items = selected_top + selected_others

        if len(rec_items) < top_n:
            target_info = product_details[product_details["product_id"] == target_product_id]
            if not target_info.empty:
                target_category = target_info["category_code"].iloc[0]
                target_brand = target_info["brand"].iloc[0]
                target_name = target_info["name"].iloc[0]

                similar_products = product_details[
                    ((product_details["category_code"] == target_category) &
                     (product_details["brand"] == target_brand)) &
                    (product_details["product_id"] != target_product_id) &
                    (~product_details["product_id"].isin(rec_items))
                    ].drop_duplicates(subset=["product_id"])

                logging.info(f'similar_products: {len(similar_products)}')

                if not similar_products.empty and len(rec_items) < top_n:
                    needed = top_n - len(rec_items)
                    tfidf = TfidfVectorizer(stop_words='english')
                    all_names = [str(target_name)] + similar_products["name"].fillna('').tolist()
                    tfidf_matrix = tfidf.fit_transform(all_names)
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

                    sim_df = pd.DataFrame({
                        'product_id': similar_products['product_id'].values,
                        'similarity': cosine_sim
                    })

                    additional_items = sim_df.sort_values('similarity', ascending=False)[
                        'product_id'
                    ].head(needed).tolist()

                    rec_items.extend(additional_items)

        recommendations = pd.DataFrame({"product_id": rec_items})
        recommendations = recommendations.merge(product_details, on="product_id", how="left")
        return recommendations.head(top_n)

# Global variable to hold the current cold start cluster.
CURRENT_COLD_CLUSTER1 = None


def recommend_products_anonymous(cold_start_clusters,df, target_product_id):


    product_details = df[["product_id", "category_code", "brand", "name"]].drop_duplicates(subset=["product_id"])
    logging.info("User not found in SVD model. Using Cold-Start Recommendation.")
    type_target_product_id = type(target_product_id)
    logging.info(f'type of target_product_id:{type_target_product_id }')
    target_product_id = str(target_product_id)
    logging.info(f'target_product_id:{target_product_id}')

    # Update CURRENT_COLD_CLUSTER
    global CURRENT_COLD_CLUSTER1
    if (CURRENT_COLD_CLUSTER1 is None or
            (target_product_id not in CURRENT_COLD_CLUSTER1.get("top", []) and
             target_product_id not in CURRENT_COLD_CLUSTER1.get("others", []))):
        if target_product_id in cold_start_clusters:
            CURRENT_COLD_CLUSTER1 = cold_start_clusters[target_product_id]
            logging.info(f'CURRENT_COLD_CLUSTER:{CURRENT_COLD_CLUSTER1}')
        else:
            for clust in cold_start_clusters.values():
                if (target_product_id in clust.get("top", []) or
                        target_product_id in clust.get("others", [])):
                    CURRENT_COLD_CLUSTER1 = clust
                    break

    # Get cluster data
    cluster = CURRENT_COLD_CLUSTER1
    top_candidates = cluster.get("top", [])
    others_candidates = cluster.get("others", [])

    # Filter out target_product_id and select candidates
    selected_top = [pid for pid in top_candidates[:8] if pid != target_product_id]
    filtered_others = [pid for pid in others_candidates if pid != target_product_id]
    selected_others = random.sample(filtered_others, 2) if len(filtered_others) >= 2 else filtered_others
    rec_items = selected_others + selected_top

    # Check if total recommendations are less than 10
    min_rec_items = 10
    if len(rec_items) < min_rec_items:
        # Get target product info
        target_info = product_details[product_details["product_id"] == target_product_id]
        if not target_info.empty:
            target_category = target_info["category_code"].iloc[0]
            target_brand = target_info["brand"].iloc[0]
            target_name = target_info["name"].iloc[0]

            # Find similar products by category_code and brand
            similar_products = product_details[
                ((product_details["category_code"] == target_category) &
                 (product_details["brand"] == target_brand)) &
                (product_details["product_id"] != target_product_id) &
                (~product_details["product_id"].isin(rec_items))
                ].drop_duplicates(subset=["product_id"])

            logging.info(f'similar_products: {len(similar_products)}')

            # If still need more items, use cosine similarity on product names
            needed = min_rec_items - len(rec_items)
            if not similar_products.empty and needed > 0:
                # Prepare TF-IDF vectors
                tfidf = TfidfVectorizer(stop_words='english')
                # Combine target name with similar products' names
                all_names = [str(target_name)] + similar_products["name"].fillna('').tolist()
                tfidf_matrix = tfidf.fit_transform(all_names)

                # Calculate cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

                # Create DataFrame with similarity scores
                sim_df = pd.DataFrame({
                    'product_id': similar_products['product_id'].values,
                    'similarity': cosine_sim
                })

                # Sort by similarity and get top items
                additional_items = sim_df.sort_values('similarity', ascending=False)[
                    'product_id'
                ].head(needed).tolist()

                rec_items.extend(additional_items)

    # Create recommendations DataFrame
    recommendations = pd.DataFrame({"product_id": rec_items})
    recommendations = recommendations.merge(product_details, on="product_id", how="left")
    return recommendations
