import pickle
import random

def train_cold_start_clusters(df, output_file="cold_start.pkl"):
    # Group sessions: each session gives a list of unique product ids.
    session_groups = df.groupby("user_session")["product_id"].apply(lambda x: list(set(x)))

    # Build candidate frequency counts for each target product.
    cluster_candidates = {}
    for products in session_groups:
        for target in products:
            cluster_candidates.setdefault(target, {})
            for candidate in products:
                if candidate == target:
                    continue
                cluster_candidates[target][candidate] = cluster_candidates[target].get(candidate, 0) + 1

    # Build clusters: for each target, sort candidates by frequency and select 8 top and 4 random from remaining.
    cold_start_clusters = {}
    for target, candidates in cluster_candidates.items():
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        sorted_candidate_ids = [cid for cid, freq in sorted_candidates]
        top_candidates = sorted_candidate_ids[:8]
        remaining = sorted_candidate_ids[8:]
        other_candidates = random.sample(remaining, 5) if len(remaining) >= 5 else remaining
        cold_start_clusters[target] = {"top": top_candidates, "others": other_candidates}

    with open(output_file, "wb") as f:
        pickle.dump(cold_start_clusters, f)
    print(f"Cold start clusters saved to {output_file}")

