"""Load MovieLens 1M, inspect it, and split positive histories.

Run from the repository root:
    python -m src.data --data-dir data/raw/ml-1m

Original movie IDs and complete raw ratings are retained for downstream
dataset construction and candidate filtering.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_integer_dtype

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


@dataclass(frozen=True)
class Interaction:
    """A positive movie interaction, retaining its rating timestamp."""

    movie_id: int
    timestamp: int


@dataclass(frozen=True)
class UserSplit:
    """One user's training observations and two held-out targets."""

    user_id: int
    train: tuple[Interaction, ...]
    validation: Interaction
    test: Interaction


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Reject malformed data instead of silently changing the experiment."""
    missing = set(RATING_COLUMNS) - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    if ratings.empty:
        raise ValueError("The ratings table is empty.")
    if ratings[RATING_COLUMNS].isna().any().any():
        raise ValueError("Ratings contain missing values.")
    if not all(is_integer_dtype(ratings[c]) for c in RATING_COLUMNS):
        raise ValueError("MovieLens 1M rating columns must contain integers.")
    if (ratings[["userId", "movieId"]] <= 0).any().any():
        raise ValueError("User and movie IDs must be positive.")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be whole numbers from 1 to 5.")
    if (ratings["timestamp"] < 0).any():
        raise ValueError("Timestamps must be nonnegative.")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError(
            "Duplicate user-movie ratings found. Agree on a handling rule "
            "before splitting; do not silently drop them."
        )


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ratings and metadata separately, preserving low ratings too."""
    data_dir = Path(data_dir)
    for filename in ("ratings.dat", "movies.dat"):
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(f"Missing file: {data_dir / filename}")

    # There are no header rows. The two-character separator needs the Python
    # parser. QUOTE_NONE preserves literal quote marks in movie titles.
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        dtype="int64",
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
    )
    if ratings.shape[1] != 4:
        raise ValueError("ratings.dat must have exactly four columns.")
    ratings.columns = RATING_COLUMNS
    validate_ratings(ratings)

    # Latin-1 also handles accented characters in the original movie file.
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    if movies.shape[1] != 3:
        raise ValueError("movies.dat must have exactly three columns.")
    movies.columns = ["movieId", "title", "genres"]
    if not is_integer_dtype(movies["movieId"]):
        raise ValueError("Movie metadata IDs must be integers.")
    if (movies["movieId"] <= 0).any() or movies["movieId"].duplicated().any():
        raise ValueError("Movie metadata IDs must be unique and positive.")
    if movies[["title", "genres"]].isna().any().any():
        raise ValueError("Movie metadata contains missing values.")
    if movies[["title", "genres"]].eq("").any().any():
        raise ValueError("Movie metadata contains empty titles or genres.")
    if not ratings["movieId"].isin(movies["movieId"]).all():
        raise ValueError("Some rated movie IDs have no metadata.")

    return ratings, movies


def split_positive_histories(
    ratings: pd.DataFrame,
    positive_threshold: int = 4,
    min_positive: int = 5,
) -> dict[int, UserSplit]:
    """Hold out the last two positives separately for each eligible user.

    Ties are ordered by movie ID. This convention is reproducible but does
    not establish the real viewing order of simultaneously rated movies.
    """
    validate_ratings(ratings)
    if not 1 <= positive_threshold <= 5:
        raise ValueError("positive_threshold must be between 1 and 5.")
    if min_positive < 4:
        raise ValueError("min_positive must be >= 4 to leave a training pair.")

    # A copy protects the complete raw table needed later for exclusions.
    positives = ratings.loc[ratings["rating"] >= positive_threshold].copy()
    positives = positives.sort_values(["userId", "timestamp", "movieId"])

    splits = {}
    for user_id, group in positives.groupby("userId", sort=True):
        if len(group) < min_positive:
            continue
        events = tuple(
            Interaction(movie_id=int(row.movieId), timestamp=int(row.timestamp))
            for row in group.itertuples(index=False)
        )
        splits[int(user_id)] = UserSplit(
            user_id=int(user_id),
            train=events[:-2],
            validation=events[-2],
            test=events[-1],
        )
    return splits


def summarize_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    splits: dict[int, UserSplit],
    positive_threshold: int,
) -> dict:
    """Small, serializable statistics for checking the chosen preprocessing."""
    positives = ratings.loc[ratings["rating"] >= positive_threshold]
    lengths = positives.groupby("userId").size()
    return {
        "ratings": len(ratings),
        "users": int(ratings["userId"].nunique()),
        "movies_with_ratings": int(ratings["movieId"].nunique()),
        "movies_in_metadata": len(movies),
        "positive_threshold": positive_threshold,
        "positive_interactions": len(positives),
        "eligible_users": len(splits),
        "excluded_users": int(ratings["userId"].nunique()) - len(splits),
        "positive_history_quantiles_before_user_filter": {
            str(q): float(lengths.quantile(q)) for q in (0, 0.25, 0.5, 0.75, 1)
        } if not lengths.empty else {},
        "ratings_in_tied_user_timestamp_groups": int(
            ratings.duplicated(["userId", "timestamp"], keep=False).sum()
        ),
        "training_positive_interactions": sum(len(s.train) for s in splits.values()),
        "training_prefix_examples": sum(len(s.train) - 1 for s in splits.values()),
        "validation_targets_before_catalogue_filter": len(splits),
        "test_targets_before_catalogue_filter": len(splits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--positive-threshold", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--min-positive", type=int, default=5)
    args = parser.parse_args()

    ratings, movies = load_movielens(args.data_dir)
    splits = split_positive_histories(
        ratings, args.positive_threshold, args.min_positive
    )
    print(json.dumps(summarize_data(
        ratings, movies, splits, args.positive_threshold
    ), indent=2))
    print("\nSplit settings:", {
        "min_positive": args.min_positive,
        "ordering": ["timestamp", "movieId"],
    })
    if splits:
        sample = next(iter(splits.values()))
        print("\nExample user:", sample.user_id)
        print("Last training movies:", [e.movie_id for e in sample.train[-5:]])
        print("Validation movie:", sample.validation.movie_id)
        print("Test movie:", sample.test.movie_id)
    else:
        print("\nNo users meet the chosen positive-history requirement.")
    print("\nOriginal movie IDs are retained; catalogue filtering is a later step.")


if __name__ == "__main__":
    main()
