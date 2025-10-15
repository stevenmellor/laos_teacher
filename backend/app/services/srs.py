"""Spaced repetition scheduling utilities."""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("SRS service module loaded")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    lao_text TEXT NOT NULL,
    romanised TEXT NOT NULL,
    translation TEXT,
    level TEXT NOT NULL,
    tag TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    reviewed_at TEXT NOT NULL,
    ease REAL NOT NULL,
    interval INTEGER NOT NULL,
    next_due TEXT NOT NULL
);
"""


@dataclass
class ReviewLog:
    card_id: str
    reviewed_at: dt.datetime
    ease: float
    interval: int
    next_due: dt.datetime


class SrsRepository:
    """Simple SM-2 style scheduler backed by SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()
        logger.info("SRS repository initialised", extra={"db_path": str(self.db_path)})

    def _connect(self) -> sqlite3.Connection:
        logger.debug("Opening SQLite connection", extra={"db_path": str(self.db_path)})
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        logger.debug("Ensured SRS schema", extra={"db_path": str(self.db_path)})

    def upsert_card(
        self,
        card_id: str,
        lao_text: str,
        romanised: str,
        translation: Optional[str],
        level: str,
        tag: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cards(id, lao_text, romanised, translation, level, tag)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    lao_text=excluded.lao_text,
                    romanised=excluded.romanised,
                    translation=excluded.translation,
                    level=excluded.level,
                    tag=excluded.tag
                """,
                (card_id, lao_text, romanised, translation, level, tag),
            )
            conn.commit()
        logger.info(
            "Card upserted",
            extra={"card_id": card_id, "level": level, "tag": tag},
        )

    def log_review(self, card_id: str, ease: float, prev_interval: Optional[int] = None) -> ReviewLog:
        now = dt.datetime.utcnow()
        if prev_interval is None:
            prev_interval = 1
        interval = max(1, int(prev_interval * (1 + ease)))
        next_due = now + dt.timedelta(days=interval)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reviews(card_id, reviewed_at, ease, interval, next_due)
                VALUES(?, ?, ?, ?, ?)
                """,
                (card_id, now.isoformat(), ease, interval, next_due.isoformat()),
            )
            conn.commit()
        logger.info(
            "Review logged",
            extra={"card_id": card_id, "interval": interval, "ease": ease},
        )
        return ReviewLog(card_id=card_id, reviewed_at=now, ease=ease, interval=interval, next_due=next_due)

    def due_cards(self, limit: int = 10) -> Iterable[str]:
        today = dt.datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT card_id FROM reviews
                WHERE next_due <= ?
                ORDER BY next_due ASC
                LIMIT ?
                """,
                (today, limit),
            )
            rows = cursor.fetchall()
        due_list = [row[0] for row in rows]
        logger.debug("Fetched due cards", extra={"count": len(due_list), "limit": limit})
        return due_list


__all__ = ["SrsRepository", "ReviewLog"]
