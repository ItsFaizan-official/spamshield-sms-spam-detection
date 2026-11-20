"""SQLite storage for locally recorded SpamShield predictions."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "spamshield.db"


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create the prediction-history table if it does not exist."""
    with closing(get_connection(db_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_message TEXT NOT NULL,
                predicted_category TEXT NOT NULL
                    CHECK (predicted_category IN ('ham', 'spam')),
                model_name TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(
    message: str,
    category: str,
    model_name: str,
    db_path: Path | str = DB_PATH,
) -> None:
    """Save one classified message to the local database."""
    if category not in {"ham", "spam"}:
        raise ValueError("category must be either 'ham' or 'spam'")
    with closing(get_connection(db_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO predictions (original_message, predicted_category, model_name)
            VALUES (?, ?, ?)
            """,
            (message, category, model_name),
        )


def get_prediction_history(
    limit: int = 10,
    db_path: Path | str = DB_PATH,
) -> list[dict[str, object]]:
    """Return the newest saved predictions first."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    with closing(get_connection(db_path)) as connection, connection:
        rows = connection.execute(
            """
            SELECT prediction_id, original_message, predicted_category,
                   model_name, prediction_timestamp
            FROM predictions
            ORDER BY prediction_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
