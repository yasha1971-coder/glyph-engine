#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import glyph_http_query_v1


class TestHTTPQueryProtocolV1(unittest.TestCase):

    def test_http_query_returns_protocol_json(self):
        with TestClient(glyph_http_query_v1.app) as client:

            health = client.get("/health")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(
                health.json(),
                {"status": "ok"},
            )

            resp = client.post(
                "/query",
                json={
                    "pattern_hex": "6572726f72"
                },
            )

            self.assertEqual(resp.status_code, 200)

            obj = resp.json()

            self.assertEqual(obj["pattern_hex"], "6572726f72")
            self.assertEqual(obj["interval"], [20, 22])
            self.assertEqual(obj["count"], 2)
            self.assertEqual(obj["fm_version"], "FMBINv2")
            self.assertEqual(obj["verified"], True)

    def test_http_query_batch_returns_protocol_json(self):
        with TestClient(glyph_http_query_v1.app) as client:

            resp = client.post(
                "/query_batch",
                json={
                    "patterns": [
                        "6572726f72",
                        "343034",
                    ]
                },
            )

            self.assertEqual(resp.status_code, 200)

            obj = resp.json()

            self.assertEqual(obj["fm_version"], "FMBINv2")

            self.assertEqual(
                obj["results"],
                [
                    {
                        "pattern_hex": "6572726f72",
                        "interval": [20, 22],
                        "count": 2,
                        "verified": True,
                    },
                    {
                        "pattern_hex": "343034",
                        "interval": [16, 17],
                        "count": 1,
                        "verified": True,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
