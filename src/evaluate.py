"""Offline evaluation: Popularity vs NLP Content-Based vs User-Based CF vs Hybrid.

Protocol: for every visitor, hide N_TEST ratings (test set), train on the rest,
rank unvisited exhibits, and check how many of the top-5 are hidden items the visitor liked (rating >= 4).
Repeated over many random splits and averaged, because the dataset is small.

Run from the project root:  python src/evaluate.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlp_pipeline import NLPPipeline
from recommender import Recommender, load_data, METHODS

N_TEST = 3          # hidden ratings per visitor
N_SEEDS = 20        # number of random splits
K = 5               # Precision@K
LIKE = 4            # rating >= LIKE counts as relevant
RESULTS_DIR = "results"
LABELS = {"popularity": "Popularity", "content": "NLP Content", "cf": "User-based CF", "hybrid": "Hybrid"}


def split_ratings(ratings, n_test=N_TEST, seed=0):
    """Hide n_test random ratings per user."""
    test = pd.concat([g.sample(min(n_test, len(g) - 4), random_state=seed)
                      for _, g in ratings.groupby("user_id")])
    return ratings.drop(test.index), test


def evaluate_split(items, ratings, seed):
    train, test = split_ratings(ratings, seed=seed)
    rec = Recommender(NLPPipeline().fit(items), train)
    out = {}
    for m in METHODS:
        precisions, sq_err, abs_err = [], [], []
        for user, g in test.groupby("user_id"):
            top = set(rec.recommend(user, m, K)["item_id"])
            relevant = set(g.loc[g["rating"] >= LIKE, "item_id"])
            precisions.append(len(top & relevant) / K)
            for item, r in zip(g["item_id"], g["rating"]):
                e = rec.predict_rating(user, item, m) - r
                sq_err.append(e ** 2)
                abs_err.append(abs(e))
        out[m] = {"precision@5": np.mean(precisions),
                  "rmse": np.sqrt(np.mean(sq_err)), "mae": np.mean(abs_err)}
    return out


def run_evaluation(items, ratings, n_seeds=N_SEEDS):
    rows = []
    for seed in range(n_seeds):
        for m, metrics in evaluate_split(items, ratings, seed).items():
            rows.append({"seed": seed, "method": m, **metrics})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- charts
def plot_rating_distribution(items, ratings, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = ratings["rating"].value_counts().reindex(range(1, 6), fill_value=0)
    ax[0].bar(counts.index, counts.values, color="#4C72B0")
    ax[0].set_title("Rating distribution")
    ax[0].set_xlabel("Rating")
    ax[0].set_ylabel("Number of ratings")
    merged = ratings.merge(items[["item_id", "gallery_wing"]], on="item_id")
    wing = merged.groupby("gallery_wing")["rating"].agg(["mean", "count"]).sort_values("mean")
    ax[1].barh(wing.index, wing["mean"], color="#55A868")
    ax[1].set_title("Average rating by gallery")
    ax[1].set_xlabel("Average rating (1-5)")
    ax[1].set_xlim(0, 5)
    for i, (m, c) in enumerate(zip(wing["mean"], wing["count"])):
        ax[1].text(m + 0.05, i, f"n={c}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_method_comparison(summary, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [LABELS[m] for m in summary.index]
    colors = ["#999999", "#4C72B0", "#DD8452", "#55A868"]
    ax[0].bar(names, summary[("precision@5", "mean")], yerr=summary[("precision@5", "std")],
              color=colors, capsize=4)
    ax[0].set_title(f"Precision@{K} (higher is better)")
    ax[0].set_ylabel("Precision@5")
    ax[1].bar(names, summary[("rmse", "mean")], yerr=summary[("rmse", "std")], color=colors, capsize=4)
    ax[1].set_title("RMSE (lower is better)")
    ax[1].set_ylabel("RMSE")
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    fig.suptitle(f"Method comparison (mean +/- std over {N_SEEDS} random splits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    items, ratings = load_data()

    n_users, n_items = ratings["user_id"].nunique(), items["item_id"].nunique()
    sparsity = 1 - len(ratings) / (n_users * n_items)
    print(f"Data: {n_items} items, {n_users} users, {len(ratings)} ratings, sparsity {sparsity:.1%}")

    plot_rating_distribution(items, ratings, f"{RESULTS_DIR}/rating_distribution.png")

    df = run_evaluation(items, ratings)
    df.to_csv(f"{RESULTS_DIR}/evaluation_raw.csv", index=False)
    summary = df.groupby("method")[["precision@5", "rmse", "mae"]].agg(["mean", "std"]).reindex(list(METHODS))
    plot_method_comparison(summary, f"{RESULTS_DIR}/method_comparison.png")

    table = pd.DataFrame({
        "Method": [LABELS[m] for m in summary.index],
        "Precision@5": summary[("precision@5", "mean")].round(3).values,
        "RMSE": summary[("rmse", "mean")].round(3).values,
        "MAE": summary[("mae", "mean")].round(3).values,
    })
    table.to_csv(f"{RESULTS_DIR}/evaluation_summary.csv", index=False)
    print(f"\nResults over {N_SEEDS} random splits ({N_TEST} hidden ratings per visitor, relevant = rating >= {LIKE}):")
    print(table.to_string(index=False))
    print(f"\nSaved charts and tables to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()