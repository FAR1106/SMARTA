import unittest
from unittest.mock import patch

from fastapi import HTTPException

import api


class FrontendSupportApiTests(unittest.TestCase):
    def test_health_reports_each_connection_source_separately(self):
        with (
            patch.object(api, "corpus_stats", return_value={
                "documents": 12,
                "maintenance_records": 3,
                "sources": 2,
            }),
            patch.object(api._predictor, "metadata", return_value={}),
            patch.object(api, "mysql_credentials_available", return_value=False),
            patch.object(api, "_ollama_model_status", return_value="ready"),
        ):
            result = api.health()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["corpus"]["maintenance_records"], 3)
        self.assertEqual(result["ollama_status"], "ready")
        self.assertFalse(result["mysql_credentials_configured"])
        self.assertEqual(result["sensor_status"], "not_connected")

    def test_maintenance_history_returns_local_rows_and_pagination(self):
        expected = {
            "total": 1,
            "limit": 5,
            "offset": 10,
            "items": [{"machine": "SMT13", "action": "Cleaned filter"}],
        }
        with patch.object(api, "list_maintenance_records", return_value=expected) as read:
            result = api.maintenance_history(limit=5, offset=10)

        self.assertEqual(result, expected)
        read.assert_called_once_with(limit=5, offset=10)

    def test_maintenance_history_hides_storage_exception_details(self):
        with patch.object(
            api,
            "list_maintenance_records",
            side_effect=OSError("private database path"),
        ):
            with self.assertRaises(HTTPException) as raised:
                api.maintenance_history(limit=5, offset=0)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            "Local maintenance history is unavailable.",
        )
        self.assertNotIn("private database path", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
