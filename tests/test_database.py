"""Tests for SpamShield's SQLite database operations."""

import sqlite3

from src.database import get_prediction_history, init_db, save_prediction


def test_database_creation(tmp_path):
    database_path = tmp_path / "test_spamshield.db"

    init_db(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    assert ("predictions",) in tables


def test_saving_prediction(tmp_path):
    database_path = tmp_path / "test_spamshield.db"
    init_db(database_path)

    save_prediction("Your parcel is ready", "spam", "Test Model", database_path)

    history = get_prediction_history(db_path=database_path)
    assert len(history) == 1
    assert history[0]["original_message"] == "Your parcel is ready"
    assert history[0]["predicted_category"] == "spam"
    assert history[0]["model_name"] == "Test Model"


def test_retrieving_prediction_history(tmp_path):
    database_path = tmp_path / "test_spamshield.db"
    init_db(database_path)
    save_prediction("First message", "ham", "Test Model", database_path)
    save_prediction("Second message", "spam", "Test Model", database_path)

    history = get_prediction_history(limit=1, db_path=database_path)

    assert len(history) == 1
    assert history[0]["original_message"] == "Second message"
    assert history[0]["prediction_id"] == 2
