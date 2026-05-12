import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
SAMPLE_LIMIT = 50000
EXPECTED_FILES = ["device.csv", "logon.csv", "file.csv", "email.csv", "http.csv"]


def _safe_load(fname: str) -> pd.DataFrame | None:
    """Load a CSV from uploads/, sampling to SAMPLE_LIMIT rows."""
    path = os.path.join(UPLOAD_DIR, fname)
    if not os.path.exists(path):
        logger.warning(f"{fname} not found in uploads/")
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if len(df) > SAMPLE_LIMIT:
            logger.info(f"{fname}: sampling {SAMPLE_LIMIT} from {len(df)} rows")
            df = df.sample(n=SAMPLE_LIMIT, random_state=42).reset_index(drop=True)
        logger.info(f"{fname}: loaded {len(df)} rows, cols={list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {fname}: {e}")
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive)."""
    col_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    return None


def _extract_user_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and normalise the user and date columns from any CSV format.
    Handles CERT insider-threat dataset columns like 'user', 'date', 'pc', etc.
    """
    df = df.copy()

    # --- User column ---
    user_col = _find_col(df, ["user", "user_id", "userid", "UserID", "employee", "emp_id"])
    if user_col and user_col != "user":
        df.rename(columns={user_col: "user"}, inplace=True)
    elif user_col is None:
        logger.warning(f"No user column detected in columns: {list(df.columns)}")

    # --- Date column ---
    date_col = _find_col(df, ["date", "datetime", "timestamp", "time", "Date", "login_time", "logon_time"])
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # Try all columns for date-like content
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(20)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > 10:
                    df["date"] = pd.to_datetime(df[col], errors="coerce")
                    logger.info(f"Auto-detected date column: '{col}'")
                    break
            except Exception:
                continue
        else:
            df["date"] = pd.NaT

    return df


def load_and_merge() -> pd.DataFrame:
    """
    Load uploaded CSV files, extract user + date, merge by grouping
    activity counts per user per day.

    Raises ValueError if no valid CSV files are found.

    Returns DataFrame with columns:
        [user, date, activity_count, device_count, logon_count,
         file_count, email_count, http_count]
    """
    frames = []
    sources = {
        "device": "device.csv",
        "logon": "logon.csv",
        "file": "file.csv",
        "email": "email.csv",
        "http": "http.csv",
    }

    loaded_sources = []

    for source_name, fname in sources.items():
        df = _safe_load(fname)
        if df is None:
            continue
        df = _extract_user_date(df)
        if "user" not in df.columns:
            logger.warning(f"No user column found in {fname} — skipping.")
            continue
        df["source"] = source_name
        frames.append(df[["user", "date", "source"]])
        loaded_sources.append(source_name)

    if not frames:
        raise ValueError(
            "No valid CSV files found in uploads/. "
            "Please upload at least one of: " + ", ".join(EXPECTED_FILES)
        )

    logger.info(f"Loaded sources: {loaded_sources}")
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["user"], inplace=True)

    # Coerce date to date-only (day granularity)
    combined["date"] = combined["date"].dt.date
    combined.dropna(subset=["date"], inplace=True)

    # Total activity count per user per date
    activity = (
        combined.groupby(["user", "date"])
        .size()
        .reset_index(name="activity_count")
    )

    # Per-source counts
    for src in sources.keys():
        src_df = combined[combined["source"] == src]
        if src_df.empty:
            activity[f"{src}_count"] = 0
            continue
        src_counts = (
            src_df.groupby(["user", "date"]).size().reset_index(name=f"{src}_count")
        )
        activity = activity.merge(src_counts, on=["user", "date"], how="left")

    # Fill NaN source counts
    for src in sources.keys():
        col = f"{src}_count"
        if col in activity.columns:
            activity[col] = activity[col].fillna(0).astype(int)
        else:
            activity[col] = 0

    activity["date"] = pd.to_datetime(activity["date"])
    activity.sort_values(["user", "date"], inplace=True)
    activity.reset_index(drop=True, inplace=True)

    n_users = activity["user"].nunique()
    logger.info(f"Merged data: {len(activity)} rows, {n_users} unique users")

    if n_users < 2:
        raise ValueError(
            f"Only {n_users} user(s) found after merging. "
            "Need at least 2 distinct users for the ML pipeline."
        )

    return activity
