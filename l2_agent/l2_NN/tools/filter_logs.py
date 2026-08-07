import re
import argparse
import numpy as np
import pandas as pd

# --- Combat Action Keywords ---

COMBAT_KEYWORDS = [
    "selected target",
    "attacked",
    "cast skill",
    "used skill",
    "effect applied",
    "performed action",
    "shot",
    "used item",
]

COMBAT_PATTERN = "|".join(re.escape(k) for k in COMBAT_KEYWORDS)


def filter_combat_data(
    df: pd.DataFrame, combat_timeout_sec: float = 5.0
) -> pd.DataFrame:
    """Filters the DataFrame to keep only combat-related actions and surrounding context window."""
    if df.empty:
        return df

    # Text-based combat action detection
    is_combat_text = df["action_raw"].str.contains(
        COMBAT_PATTERN, case=False, na=False
    )

    # Boolean flags present in the Parquet schema
    is_attacking = df["is_attacking"] if "is_attacking" in df.columns else False
    is_casting = df["is_casting"] if "is_casting" in df.columns else False
    in_combat_flag = df["in_combat"] if "in_combat" in df.columns else False

    # Mark active combat trigger rows
    is_combat_action = is_combat_text | is_attacking | is_casting | in_combat_flag

    # Timestamp conversion for exact window calculations
    dt = pd.to_datetime(df["timestamp"], format="%Y.%m.%d %H:%M:%S,%f")

    # Time window calculation (pure Pandas approach without NumPy warnings)
    last_combat_time = dt.where(is_combat_action).ffill()

    time_since_combat = (dt - last_combat_time).dt.total_seconds()

    # Mask to keep combat actions and events within the timeout window
    combat_mask = is_combat_action | (time_since_combat <= combat_timeout_sec)

    return df[combat_mask].copy()


def process_filter_parquet(input_path: str, output_path: str, timeout: float):
    print(f"Starting to read file: {input_path}")

    df = pd.read_parquet(input_path, engine="pyarrow")
    initial_count = len(df)
    print(f"Total initial records: {initial_count}")

    print(f"Filtering combat data with timeout = {timeout} seconds...")
    filtered_df = filter_combat_data(df, combat_timeout_sec=timeout)
    filtered_count = len(filtered_df)

    removed_count = initial_count - filtered_count
    reduction_pct = (
        (removed_count / initial_count * 100) if initial_count > 0 else 0
    )

    print(f"Successfully filtered {filtered_count} combat action frames in total.")
    print(
        f"Removed {removed_count} non-combat noise records ({reduction_pct:.2f}% reduction)."
    )

    filtered_df.to_parquet(output_path, index=False, compression="snappy")
    print(f"Successfully saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter non-combat noise from player action Parquet file."
    )
    parser.add_argument("input_parquet", type=str, help="Path to the input Parquet file.")
    parser.add_argument("output_parquet", type=str, help="Path to the output Parquet file.")
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=5.0,
        help="Combat timeout window in seconds (default: 5.0).",
    )

    args = parser.parse_args()

    process_filter_parquet(args.input_parquet, args.output_parquet, args.timeout)
