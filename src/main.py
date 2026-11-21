"""Demo entry point for the Museum Exhibit Recommender.

Run from the project root:
    python src/main.py                              # run the 3 required test cases -> results/sample_outputs.txt
    python src/main.py --user U07                   # top-5 unvisited exhibits for a visitor (hybrid)
    python src/main.py --user U07 --method cf       # popularity | content | cf | hybrid
    python src/main.py --query "silk road textiles" # free-text query
    python src/main.py --similar E08                # exhibits similar to one exhibit
"""
import argparse
import os
import pandas as pd

from nlp_pipeline import NLPPipeline
from recommender import Recommender, load_data, METHODS

THEME_QUERY = "ancient maritime trade, traditional ceramics, and maps"
RETURNING_USER = "U07"
OUTPUT_FILE = "results/sample_outputs.txt"

# Newly opened temporary exhibit: placard text only, no ratings (cold-start test)
NEW_EXHIBIT = {
    "item_id": "E39",
    "exhibit_title": "Temporary: Cargo of the Ca Mau Shipwreck",
    "gallery_wing": "Special Exhibitions",
    "era": "17th century",
    "placard_text": ("Blue-and-white porcelain bowls and sea charts recovered from a trading ship that sank "
                     "in the South China Sea. The finds show how ceramics and maps travelled together "
                     "on merchant routes."),
    "theme_tags": "maritime, ceramics, maps, shipwreck, trade",
}

LOG = []


def emit(text=""):
    """Print and remember a line, so the same text can be saved to sample_outputs.txt."""
    print(text)
    LOG.append(text)


def show(df):
    emit(df.to_string(index=False) if len(df) else "(no results)")


def build():
    items, ratings = load_data()
    nlp = NLPPipeline().fit(items)
    return items, ratings, Recommender(nlp, ratings)


# ------------------------------------------------------------------ test cases
def test_case_1(rec):
    emit("=" * 78)
    emit("TEST CASE 1 - Theme query")
    emit(f'Query: "{THEME_QUERY}"')
    emit("=" * 78)
    show(rec.recommend_query(THEME_QUERY, n=5))
    emit()


def test_case_2(rec, ratings):
    user = RETURNING_USER
    emit("=" * 78)
    emit(f"TEST CASE 2 - Returning visitor ({user})")
    emit("=" * 78)
    past = ratings[ratings.user_id == user].sort_values("rating", ascending=False)
    fav = past[past.rating >= 4]
    emit(f"{user} has rated {len(past)} exhibits. Past favourites (rating >= 4):")
    for _, r in fav.iterrows():
        emit(f"  {r.item_id}  {rec.title[r.item_id]}  (rating {r.rating})")
    emit()
    visited = set(past.item_id)
    for m in ("popularity", "content", "cf", "hybrid"):
        emit(f"--- {m} ---")
        out = rec.recommend(user, m, 5)
        show(out)
        emit(f"Check: none already visited -> {not (set(out.item_id) & visited)}")
        emit()


def test_case_3(rec):
    emit("=" * 78)
    emit("TEST CASE 3 - Cold-start: new temporary exhibit (placard text only, 0 ratings)")
    emit(f'New exhibit: {NEW_EXHIBIT["item_id"]} - {NEW_EXHIBIT["exhibit_title"]}')
    emit("=" * 78)
    new_id = NEW_EXHIBIT["item_id"]

    before = rec.recommend_query(THEME_QUERY, n=5)
    emit(f"Before adding it, {new_id} in top-5 for the theme query: {new_id in set(before.item_id)}")

    rec.add_new_items(pd.DataFrame([NEW_EXHIBIT]))   # uses the SAME fitted vocabulary, no refit
    emit(f"\nAfter adding it (theme query: \"{THEME_QUERY}\"):")
    after = rec.recommend_query(THEME_QUERY, n=5)
    show(after)
    rank = after.index[after.item_id == new_id]
    emit(f"Check: {new_id} discovered in top-5 -> {len(rank) > 0}"
         + (f" (rank {after.loc[rank[0], 'rank']})" if len(rank) else ""))

    emit(f"\nHybrid recommendations for {RETURNING_USER} with the new exhibit in the catalogue:")
    hy = rec.recommend(RETURNING_USER, "hybrid", 5)
    show(hy)
    emit(f"Check: {new_id} in {RETURNING_USER}'s hybrid top-5 -> {new_id in set(hy.item_id)}")

    pop = rec.recommend(RETURNING_USER, "popularity", 5)
    emit(f"\nPopularity baseline cannot see it (no ratings yet): {new_id} in top-5 -> {new_id in set(pop.item_id)}")

    emit(f"\nExhibits most similar to {new_id} (item-to-item, content only):")
    show(rec.similar_items(new_id, n=3))
    emit()


def run_all():
    items, ratings, rec = build()
    test_case_1(rec)
    test_case_2(rec, ratings)
    test_case_3(rec)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"Saved to {OUTPUT_FILE}")


# ------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description="Museum exhibit recommender demo")
    p.add_argument("--user", help="visitor id, e.g. U07")
    p.add_argument("--method", default="hybrid", choices=METHODS)
    p.add_argument("--query", help="free-text query, e.g. 'silk road textiles'")
    p.add_argument("--similar", help="exhibit id for item-to-item recommendations, e.g. E08")
    p.add_argument("-n", type=int, default=5, help="number of recommendations")
    args = p.parse_args()

    if not (args.user or args.query or args.similar):
        run_all()
        return

    items, ratings, rec = build()
    if args.query:
        emit(f'Top-{args.n} for query: "{args.query}"')
        show(rec.recommend_query(args.query, args.n))
    if args.user:
        if args.user not in set(ratings.user_id):
            emit(f"Unknown visitor {args.user}: showing popularity ranking instead.")
        emit(f"Top-{args.n} unvisited exhibits for {args.user} ({args.method}):")
        show(rec.recommend(args.user, args.method, args.n))
    if args.similar:
        if args.similar not in rec.idx:
            emit(f"Unknown exhibit id {args.similar}")
        else:
            emit(f"Exhibits similar to {args.similar} ({rec.title[args.similar]}):")
            show(rec.similar_items(args.similar, args.n))


if __name__ == "__main__":
    main()