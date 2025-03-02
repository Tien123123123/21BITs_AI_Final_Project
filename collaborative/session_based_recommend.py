import pandas as pd
import pickle
import random
import logging

# Global variable to hold the current cold start cluster.
CURRENT_COLD_CLUSTER = None


def recommend_products(cf_model, cold_start_clusters, df, target_user_id, target_product_id, top_n=10):

    df["event_time"] = pd.to_datetime(df["event_time"])


    product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])

    user_exists = target_user_id in cf_model.trainset._raw2inner_id_users

    if user_exists:
        # Try SVD-based recommendation for known users.


        logging.info("User exist. CF session based recommend.")

        target_max_date = df.loc[df["product_id"] == target_product_id, "event_time"].max()
        if pd.isna(target_max_date):
            logging.info(f"Warning: target_product_id={target_product_id} never appeared in the data.")
            return pd.DataFrame()

        cutoff_date = target_max_date - pd.DateOffset(months=3)
        df_recent = df[(df["event_time"] >= cutoff_date) & (df["event_time"] <= target_max_date)]
        target_sessions = df_recent.loc[df_recent["product_id"] == target_product_id, "user_session"].unique()
        related_sessions = df_recent[df_recent["user_session"].isin(target_sessions)]

        candidate_products = related_sessions["product_id"].unique().tolist()
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
        if len(recommendations) < top_n:
            additional_needed = top_n - len(recommendations)
            leftover = [pid for pid in df["product_id"].unique() if pid not in recommendations["product_id"].values]
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

        return recommendations.head(top_n)

    else:
        # SVD model failed (e.g. unknown user) – use cold start.


        logging.info("User not found in SVD model. Using Cold-Start Recommendation.")

        global CURRENT_COLD_CLUSTER
        if (CURRENT_COLD_CLUSTER is None or
                (target_product_id not in CURRENT_COLD_CLUSTER.get("top", []) and
                 target_product_id not in CURRENT_COLD_CLUSTER.get("others", []))):
            if target_product_id in cold_start_clusters:
                CURRENT_COLD_CLUSTER = cold_start_clusters[target_product_id]
            else:
                for clust in cold_start_clusters.values():
                    if (target_product_id in clust.get("top", []) or
                            target_product_id in clust.get("others", [])):
                        CURRENT_COLD_CLUSTER = clust
                        break

        cluster = CURRENT_COLD_CLUSTER
        top_candidates = cluster.get("top", [])
        others_candidates = cluster.get("others", [])

        selected_top = top_candidates[:8]
        selected_others = random.sample(others_candidates, 2) if len(others_candidates) >= 2 else others_candidates
        rec_items = selected_others + selected_top

        recommendations = pd.DataFrame({"product_id": rec_items})
        recommendations = recommendations.merge(product_details, on="product_id", how="left")
        return recommendations


# Global variable to hold the current cold start cluster.
CURRENT_COLD_CLUSTER = None


def recommend_products_anonymous(cold_start_clusters,df, target_product_id):


    product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])


    print("User not found in SVD model. Using Cold-Start Recommendation.")

    global CURRENT_COLD_CLUSTER
    if (CURRENT_COLD_CLUSTER is None or
            (target_product_id not in CURRENT_COLD_CLUSTER.get("top", []) and
             target_product_id not in CURRENT_COLD_CLUSTER.get("others", []))):
        if target_product_id in cold_start_clusters:
            CURRENT_COLD_CLUSTER = cold_start_clusters[target_product_id]
        else:
            for clust in cold_start_clusters.values():
                if (target_product_id in clust.get("top", []) or
                        target_product_id in clust.get("others", [])):
                    CURRENT_COLD_CLUSTER = clust
                    break

    cluster = CURRENT_COLD_CLUSTER
    top_candidates = cluster.get("top", [])
    others_candidates = cluster.get("others", [])

    selected_top = top_candidates[:8]
    selected_others = random.sample(others_candidates, 2) if len(others_candidates) >= 2 else others_candidates
    rec_items = selected_others + selected_top

    recommendations = pd.DataFrame({"product_id": rec_items})
    recommendations = recommendations.merge(product_details, on="product_id", how="left")
    return recommendations