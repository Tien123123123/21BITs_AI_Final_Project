import os
import sys
import pytz
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from minio_server.server import create_bucket
from minio_server.push import push_object
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

from mlxtend.frequent_patterns import fpgrowth, association_rules
from scipy.sparse import csr_matrix
import numpy as np

logging.basicConfig(level=logging.INFO)

def create_sparse_dataframe(transactions: pd.Series, *, min_item_count: int = 5) -> pd.DataFrame:
    logging.info("🔧 Creating sparse dataframe from transactions...")
    item_counts = pd.Series([item for sublist in transactions for item in sublist]).value_counts()
    frequent_items = item_counts[item_counts >= min_item_count].index
    item_to_index = {item: idx for idx, item in enumerate(frequent_items)}
    row_indices, col_indices = [], []

    for row, basket in enumerate(transactions):
        for item in basket:
            if item in item_to_index:
                row_indices.append(row)
                col_indices.append(item_to_index[item])

    sparse_matrix = csr_matrix(
        (np.ones(len(row_indices), dtype=bool), (row_indices, col_indices)),
        shape=(len(transactions), len(frequent_items)),
        dtype=bool
    )
    logging.info("✅ Sparse dataframe created.")
    return pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=frequent_items.astype(str))


def preprocess_transactions(df: pd.DataFrame, sample_ratio: float = 0.3, event_type_filter: str | None = None) -> pd.Series:
    logging.info("🔍 Preprocessing transactions...")
    if event_type_filter:
        df = df[df["event_type"] == event_type_filter]

    transactions = df.groupby("user_session")["product_id"].apply(list)
    transactions = transactions[transactions.apply(len) > 1]

    if 0 < sample_ratio < 1:
        transactions = transactions.sample(frac=sample_ratio, random_state=42)
    logging.info(f"✅ Preprocessed transactions: {len(transactions)} sessions retained.")
    return transactions


def generate_association_rules(transactions: pd.Series, min_item_count=5, min_support=0.001, min_confidence=0.08) -> pd.DataFrame:
    logging.info("▶️ Creating sparse dataframe...")
    sparse_df = create_sparse_dataframe(transactions, min_item_count=min_item_count)
    logging.info("✅ Sparse dataframe created.")
    logging.info(f"  - Shape: {sparse_df.shape}")
    logging.info(f"  - Columns: {len(sparse_df.columns)} items")
    logging.info("▶️ Running fpgrowth...")

    frequent_itemsets = fpgrowth(sparse_df, min_support=min_support, use_colnames=True)
    logging.info(f"✅ fpgrowth finished. Found {len(frequent_itemsets)} frequent itemsets.")

    logging.info("▶️ Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    logging.info(f"✅ Generated {len(rules)} rules.")

    return rules


def pretrain_association(df: pd.DataFrame, minio_bucket_name: str = "models"):
    logging.info("Starting association rule training...")

    transactions = preprocess_transactions(df, sample_ratio=0.3)
    rules = generate_association_rules(transactions)

    logging.info(f"Generated {len(rules)} rules.")

    # Save rules
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(timezone)
    formatted_time = now.strftime("%d_%m_%y_%H_%M")
    model_name = f"association_rules_{formatted_time}.pkl"

    output_path = Path("models") / model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_pickle(output_path)
    logging.info(f"Saved rules to {output_path}")

    # Upload to MinIO
    create_bucket(minio_bucket_name)
    push_object(bucket_name=minio_bucket_name, file_path=str(output_path), object_name=model_name)

    return minio_bucket_name, model_name


