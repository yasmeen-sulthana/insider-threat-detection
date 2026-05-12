import logging
import traceback
import pandas as pd
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ml.preprocessing import load_and_merge
from ml.features import normalize, create_sequences
from ml.autoencoder import train_autoencoder
from ml.bilstm import train_bilstm
from ml.classifier import (
    aggregate_user_meta_features,
    classify_users,
    build_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/run-model", tags=["model"])

# =========================================
# SIMPLE PROGRESS TRACKER
# =========================================
_progress = {
    "step": 0,
    "total": 6,
    "message": "Idle",
}


@router.post("")
async def run_model():

    global _progress

    try:

        # =========================================
        # STEP 1 — LOAD & MERGE DATA
        # =========================================
        _progress = {
            "step": 1,
            "total": 6,
            "message": "Loading & merging datasets...",
        }

        logger.info("Step 1: Loading CSV files")

        df = load_and_merge()

        # =========================================
        # FILE COUNTS
        # =========================================
        base_path = "uploads"

        csv_files = {
            "device": "device.csv",
            "logon": "logon.csv",
            "file": "file.csv",
            "email": "email.csv",
            "http": "http.csv",
        }

        file_counts = {}

        for key, filename in csv_files.items():

            path = os.path.join(base_path, filename)

            if os.path.exists(path):

                try:

                    df_temp = pd.read_csv(path)

                    # counts rows excluding header
                    file_counts[key] = len(df_temp)

                except Exception as e:

                    logger.error(f"Error reading {filename}: {e}")

                    file_counts[key] = 0

            else:

                file_counts[key] = 0

        # =========================================
        # UNIQUE USERS
        # =========================================
        unique_users = int(df["user"].nunique())

        logger.info(f"File counts: {file_counts}")
        logger.info(f"Unique users: {unique_users}")

        # =========================================
        # STEP 2 — FEATURE ENGINEERING
        # =========================================
        _progress = {
            "step": 2,
            "total": 6,
            "message": "Normalizing & creating sequences...",
        }

        logger.info("Step 2: Feature engineering")

        df_norm, scaler = normalize(df)

        user_sequences, X_all, user_labels = create_sequences(df_norm)

        if X_all.shape[0] == 0:

            return JSONResponse(
                {
                    "error": (
                        "Not enough sequence data. "
                        "Each user needs sufficient activity records."
                    )
                },
                status_code=400,
            )

        # =========================================
        # STEP 3 — AUTOENCODER
        # =========================================
        _progress = {
            "step": 3,
            "total": 6,
            "message": "Training Autoencoder...",
        }

        logger.info(
            f"Step 3: Autoencoder training on {X_all.shape[0]} sequences"
        )

        ae_errors = train_autoencoder(
            X_all,
            epochs=15,
            batch_size=64,
        )

        # =========================================
        # STEP 4 — BiLSTM
        # =========================================
        _progress = {
            "step": 4,
            "total": 6,
            "message": "Training BiLSTM...",
        }

        logger.info(
            f"Step 4: BiLSTM training on {X_all.shape[0]} sequences"
        )

        bilstm_scores = train_bilstm(
            X_all,
            ae_errors,
            epochs=15,
            batch_size=64,
        )

        # =========================================
        # STEP 5 — RANDOM FOREST
        # =========================================
        _progress = {
            "step": 5,
            "total": 6,
            "message": "Running Random Forest classifier...",
        }

        logger.info("Step 5: Aggregating meta-features")

        meta_df = aggregate_user_meta_features(
            user_sequences,
            ae_errors,
            bilstm_scores,
            user_labels,
        )

        result_df, accuracies, best_model = classify_users(meta_df)

        # =========================================
        # STEP 6 — BUILD RESPONSE
        # =========================================
        _progress = {
            "step": 6,
            "total": 6,
            "message": "Done",
        }

        logger.info("Step 6: Building response")

        summary = build_summary(
            result_df,
            accuracies,
            best_model,
        )

        # =========================================
        # ADD COUNTS
        # =========================================
        summary["file_counts"] = file_counts
        summary["unique_users"] = unique_users

        # =========================================
        # USER ACTIVITY TOTALS
        # =========================================
        user_activity = (
            df.groupby("user")["activity_count"]
            .sum()
            .reset_index()
        )

        activity_map = dict(
            zip(
                user_activity["user"],
                user_activity["activity_count"],
            )
        )

        for ur in summary["user_results"]:

            ur["activity_total"] = int(
                activity_map.get(ur["user_id"], 0)
            )

        # =========================================
        # TREND DATA
        # =========================================
        if "date" in df.columns:

            daily = (
                df.groupby("date")["activity_count"]
                .sum()
                .reset_index()
            )

            daily = daily.sort_values("date").tail(30)

            summary["trend_data"] = [
                {
                    "date": str(row["date"])[:10],
                    "activity": int(row["activity_count"]),
                }
                for _, row in daily.iterrows()
            ]

        else:

            summary["trend_data"] = []

        logger.info(
            f"Pipeline complete: "
            f"{summary['unique_users']} unique users, "
            f"{summary['threat_users']} threats, "
            f"accuracy={summary['model_accuracy']}"
        )

        return JSONResponse(summary)

    # =========================================
    # VALIDATION ERROR
    # =========================================
    except ValueError as ve:

        logger.error(f"Validation error: {ve}")

        _progress = {
            "step": 0,
            "total": 6,
            "message": "Error",
        }

        return JSONResponse(
            {"error": str(ve)},
            status_code=400,
        )

    # =========================================
    # GENERAL ERROR
    # =========================================
    except Exception as e:

        logger.error(
            f"Pipeline error:\n{traceback.format_exc()}"
        )

        _progress = {
            "step": 0,
            "total": 6,
            "message": "Error",
        }

        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


# =========================================
# PROGRESS ENDPOINT
# =========================================
@router.get("/progress")
def get_progress():

    return _progress