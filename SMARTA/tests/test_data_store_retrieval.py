import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_store


class DataStoreRetrievalTests(unittest.TestCase):
    def test_fts_reranking_keeps_relevant_passage_and_deduplicates_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "smarta.sqlite3"
            with sqlite3.connect(db_path) as db:
                db.execute(
                    "CREATE TABLE documents (id INTEGER PRIMARY KEY, source TEXT, locator TEXT, text TEXT, metadata TEXT)"
                )
                db.execute(
                    "CREATE VIRTUAL TABLE documents_fts USING fts5(text, source, locator)"
                )
                rows = [
                    (1, "sop.pdf", "page 243", "Compressor overheating maintenance checks include monitoring the air temperature switch.", "{}"),
                    (2, "sop.pdf", "page 243", "Compressor maintenance includes oil and filter checks.", "{}"),
                    (3, "manual.pdf", "page 8", "Lathe process temperature readings are recorded during machining.", "{}"),
                ]
                db.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", rows)
                db.executemany(
                    "INSERT INTO documents_fts(rowid, text, source, locator) VALUES (?, ?, ?, ?)",
                    [(row[0], row[3], row[1], row[2]) for row in rows],
                )

            with patch("data_store.DB_PATH", db_path):
                results = data_store.retrieve(
                    "What maintenance checks are supported for an overheating compressor?",
                    limit=6,
                )

        self.assertTrue(results)
        self.assertEqual((results[0]["source"], results[0]["locator"]), ("sop.pdf", "page 243"))
        self.assertIn("overheating", results[0]["text"])
        self.assertEqual(len({(item["source"], item["locator"]) for item in results}), len(results))


if __name__ == "__main__":
    unittest.main()
