from pathlib import Path
from typing import List, Sequence, Set

import pandas as pd


class CartRecommender:
    """
    Suggest complementary products given (a) an association-rules source
    – either a .pkl path or a pre-loaded DataFrame – and (b) the current
    cart items.
    """

    def __init__(self) -> None:
        self._cache: dict[str | Path, pd.DataFrame] = {}

    @staticmethod
    def _preprocess(rules: pd.DataFrame) -> pd.DataFrame:
        """Make sure antecedents/consequents are in the right shape + score."""
        rules = rules.copy()
        rules["antecedents"] = rules["antecedents"].apply(lambda s: tuple(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: next(iter(s)))
        rules["_score"] = rules["confidence"] * rules["lift"]
        return rules.sort_values("_score", ascending=False, ignore_index=True)

    def _get_rules(self, src: str | Path | pd.DataFrame) -> pd.DataFrame:
        """Return a *pre-processed* rules DataFrame, loading/caching if needed."""
        if isinstance(src, (str, Path)):
            # 1) load from pickle path
            if src not in self._cache:
                self._cache[src] = self._preprocess(pd.read_pickle(src))
            return self._cache[src]
        elif isinstance(src, pd.DataFrame):
            # 2) caller already has the DataFrame in memory
            #    (don’t cache – that’s up to the caller)
            return self._preprocess(src)
        else:
            raise TypeError(
                "rules must be a pickle path or a pandas.DataFrame "
                f"— got {type(src)!r}"
            )

    def recommend(
        self,
        rules_src: str | Path | pd.DataFrame,
        cart_items: Sequence[str | int],
        k: int = 5,
    ) -> List[str]:
        """Generate k product recommendations based on cart items, formatted as 'product_id: <id>'."""
        rules = self._get_rules(rules_src)

        cart: Set[str] = {str(x) for x in cart_items}

        # rules whose antecedents ⊆ cart
        matching = rules[rules["antecedents"].apply(lambda ant: set(ant).issubset(cart))]
        matching = matching[~matching["consequents"].isin(cart)]  # exclude in-cart

        recs = [
            f"product_id: {product}"
            for product in matching.drop_duplicates("consequents")
                                   .head(k)["consequents"].tolist()
        ]

        # fallback: top-lift consequents not in cart/recs
        if len(recs) < k:
            fallback_pool = (
                rules.drop_duplicates("consequents")
                     .head(200)["consequents"]
            )
            for prod in fallback_pool:
                formatted_prod = f"product_id: {prod}"
                if prod not in cart and formatted_prod not in recs:
                    recs.append(formatted_prod)
                    if len(recs) == k:
                        break
        return recs