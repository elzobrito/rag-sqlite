#!/usr/bin/env python3
"""Offline tests for rag_sqlite.py (no Ollama required)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "rag_sqlite.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Import pure functions without running CLI
sys.path.insert(0, str(ROOT))
import rag_sqlite as rag  # noqa: E402


class TestScoring(unittest.TestCase):
    def test_cosine_identical_is_one(self):
        v = [1.0, 0.0, 0.0]
        self.assertEqual(rag.cosine_similarity(v, v), 1.0)

    def test_cosine_orthogonal_is_zero(self):
        self.assertEqual(rag.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_keyword_score_token_overlap(self):
        self.assertEqual(
            rag.keyword_score("data mesh functions", "data mesh is a pattern"),
            rag.round_score(2 / 3),
        )
        self.assertEqual(rag.keyword_score("data mesh", "unrelated text"), 0.0)

    def test_hybrid_score_weights(self):
        self.assertEqual(
            rag.hybrid_score(cosine=1.0, keyword=0.0, alpha=0.7), 0.7
        )
        self.assertEqual(
            rag.hybrid_score(cosine=0.0, keyword=1.0, alpha=0.7), 0.3
        )
        self.assertEqual(
            rag.hybrid_score(cosine=1.0, keyword=1.0, alpha=1.0), 1.0
        )

    def test_chunk_deterministic(self):
        text = "para um\n\ndois tres " * 50
        a = rag.chunk_text(text, max_chars=120, overlap_chars=20)
        b = rag.chunk_text(text, max_chars=120, overlap_chars=20)
        self.assertEqual(a, b)
        self.assertTrue(len(a) >= 1)

    def test_hash_embed_stable(self):
        v1 = rag.embed_hash(["hello"])[0]
        v2 = rag.embed_hash(["hello"])[0]
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), rag.HASH_EMBED_DIMS)


class TestRetrieve(unittest.TestCase):
    def _rows(self):
        return [
            {
                "id": 2,
                "document_id": 1,
                "chunk_index": 1,
                "chunk_text": "mesh roles secondary",
                "provider": "hash",
                "model": "hash-32",
                "dimensions": 3,
                "embedding": [0.9, 0.1, 0.0],
                "filename": "a.txt",
                "source_path": "/a.txt",
            },
            {
                "id": 1,
                "document_id": 1,
                "chunk_index": 0,
                "chunk_text": "Funcoes para Data Mesh equipes de dominio",
                "provider": "hash",
                "model": "hash-32",
                "dimensions": 3,
                "embedding": [0.85, 0.1, 0.0],
                "filename": "a.txt",
                "source_path": "/a.txt",
            },
            {
                "id": 3,
                "document_id": 1,
                "chunk_index": 2,
                "chunk_text": "neighbor after mesh functions",
                "provider": "hash",
                "model": "hash-32",
                "dimensions": 3,
                "embedding": [0.1, 0.9, 0.0],
                "filename": "a.txt",
                "source_path": "/a.txt",
            },
        ]

    def test_ranking_order_stable(self):
        qvec = [0.9, 0.1, 0.0]
        hits = rag.retrieve(
            query="data mesh",
            query_vector=qvec,
            rows=self._rows(),
            top_k=3,
            min_score=0.0,
            hybrid_alpha=1.0,
        )
        self.assertEqual([h["chunk_id"] for h in hits], [2, 1, 3])

    def test_expand_neighbors(self):
        qvec = [0.9, 0.1, 0.0]
        hits = rag.retrieve(
            query="mesh",
            query_vector=qvec,
            rows=self._rows(),
            top_k=1,
            min_score=0.0,
            hybrid_alpha=1.0,
        )
        expanded = rag.expand_neighbors(
            hits, self._rows(), expand_neighbors=1, query="mesh"
        )
        ids = {h["chunk_id"] for h in expanded}
        self.assertIn(2, ids)
        # neighbors of chunk_index 1 are 0 and 2
        self.assertTrue(1 in ids or 3 in ids)


def _run(db: Path, *args: str) -> tuple[int, dict]:
    cmd = [sys.executable, str(SCRIPT), "--db", str(db), "--compact", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"invalid json exit={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return proc.returncode, payload


class TestCliOffline(unittest.TestCase):
    def test_auto_create_and_config_set_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "nested" / "kb.sqlite"
            self.assertFalse(db.exists())
            code, payload = _run(
                db,
                "config",
                "set-ollama",
                "--url",
                "https://ollama.example.com",
                "--model",
                "nomic-embed-text",
                "--timeout",
                "180",
            )
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(db.exists())
            self.assertTrue(payload.get("db_created"))
            self.assertEqual(payload["settings"]["embedding_provider"], "ollama")
            self.assertEqual(
                payload["settings"]["base_url"], "https://ollama.example.com"
            )
            self.assertEqual(payload["settings"]["embedding_model"], "nomic-embed-text")
            self.assertEqual(payload["settings"]["timeout_seconds"], 180)

            code2, listed = _run(db, "config", "list")
            self.assertEqual(code2, 0)
            keys = {s["key"]: s["value"] for s in listed["settings"]}
            self.assertEqual(keys["base_url"], "https://ollama.example.com")
            self.assertFalse(listed.get("db_created"))

    def test_unknown_setting_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            code, payload = _run(db, "config", "set", "nope", "1")
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["schema_version"], "rag_sqlite.error.v1")

    def test_index_query_deterministic_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, idx = _run(db, "index", str(FIXTURES))
            self.assertEqual(code, 0)
            self.assertTrue(idx["ok"])
            self.assertGreaterEqual(idx["totals"]["indexed"], 1)

            # second index → unchanged
            code2, idx2 = _run(db, "index", str(FIXTURES))
            self.assertEqual(code2, 0)
            self.assertGreaterEqual(idx2["totals"]["unchanged"], 1)

            code_q, q1 = _run(db, "query", "data mesh domains", "--top-k", "3")
            self.assertIn(code_q, (0, 2))
            self.assertTrue(q1["ok"])
            self.assertIn("context", q1)
            self.assertTrue(q1["meta"]["deterministic"])

            code_q2, q2 = _run(db, "query", "data mesh domains", "--top-k", "3")
            self.assertEqual(q1["hits"], q2["hits"])
            self.assertEqual(q1["context"], q2["context"])

            # keyword should prefer alpha.txt over gardening
            if q1["hits"]:
                top_file = q1["hits"][0].get("filename")
                self.assertEqual(top_file, "alpha.txt")

    def test_empty_query_is_ok_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            # no docs — valid retrieval with zero hits
            code, payload = _run(db, "query", "anything")
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["hits"], [])
            self.assertEqual(payload["meta"]["hit_count"], 0)

    def test_schema_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            code, payload = _run(db, "schema")
            # schema does not need db but our helper always passes --db
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(any(c["command"] == "query" for c in payload["commands"]))

    def test_docs_list_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "index", str(FIXTURES / "alpha.txt"))
            code, docs = _run(db, "docs", "list")
            self.assertEqual(code, 0)
            self.assertEqual(docs["count"], 1)
            code_s, stats = _run(db, "stats")
            self.assertEqual(code_s, 0)
            self.assertGreaterEqual(stats["chunks"], 1)

    def test_usage_error_emits_json(self):
        """Codex P0: argparse must not break the JSON stdout contract."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            # query without text → UsageError JSON
            code, payload = _run(db, "query")
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["schema_version"], "rag_sqlite.error.v1")
            self.assertEqual(payload["error"]["type"], "UsageError")

    def test_set_ollama_atomic_rollback(self):
        """Codex P0: invalid timeout must not partially apply provider/url/model."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "base_url", "http://127.0.0.1:11434")
            _run(db, "config", "set", "embedding_model", "keep-me")

            code, payload = _run(
                db,
                "config",
                "set-ollama",
                "--url",
                "https://ollama.example.com",
                "--model",
                "nomic-embed-text",
                "--timeout",
                "0",
            )
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])

            _, listed = _run(db, "config", "list")
            keys = {s["key"]: s["value"] for s in listed["settings"]}
            self.assertEqual(keys["embedding_provider"], "hash")
            self.assertEqual(keys["base_url"], "http://127.0.0.1:11434")
            self.assertEqual(keys["embedding_model"], "keep-me")

    def test_reject_nan_hybrid_alpha(self):
        """Codex P1: NaN must not enter settings or JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            code, payload = _run(db, "config", "set", "hybrid_alpha", "nan")
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertIn("non-finite", payload["error"]["message"].lower())

    def test_read_commands_do_not_create_db(self):
        """Codex P1: stats/query must not silently create empty DB."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "missing" / "nope.sqlite"
            self.assertFalse(db.exists())
            code, payload = _run(db, "stats")
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "DB_NOT_FOUND")
            self.assertFalse(db.exists())

            code2, payload2 = _run(db, "query", "x")
            self.assertEqual(code2, 1)
            self.assertEqual(payload2["error"]["type"], "DB_NOT_FOUND")

    def test_dimension_mismatch_fails_closed(self):
        """Codex P1: mismatched embedding dims must not silent-rank by keyword."""
        rows = [
            {
                "id": 1,
                "document_id": 1,
                "chunk_index": 0,
                "chunk_text": "data mesh",
                "provider": "hash",
                "model": "hash-32",
                "dimensions": 2,
                "embedding": [1.0, 0.0],
                "filename": "a.txt",
                "source_path": "/a.txt",
            }
        ]
        with self.assertRaises(rag.CliError) as ctx:
            rag.retrieve(
                query="data mesh",
                query_vector=[1.0, 0.0, 0.0],
                rows=rows,
                top_k=1,
                min_score=0.0,
                hybrid_alpha=1.0,
            )
        self.assertEqual(ctx.exception.error_type, "DimensionMismatch")


if __name__ == "__main__":
    unittest.main()


class TestPlanFeatures(unittest.TestCase):
    def test_fingerprint_changes_with_base_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, s1 = _run(db, "stats")
            self.assertEqual(code, 0)
            fp1 = s1["index_fingerprint"]
            # hash provider ignores base_url in fingerprint (empty)
            _run(db, "config", "set", "embedding_provider", "ollama")
            _run(db, "config", "set", "base_url", "http://127.0.0.1:11434")
            _, s2 = _run(db, "stats")
            fp2 = s2["index_fingerprint"]
            _run(db, "config", "set", "base_url", "http://192.168.0.10:11434")
            _, s3 = _run(db, "stats")
            fp3 = s3["index_fingerprint"]
            self.assertNotEqual(fp2, fp3)
            self.assertNotEqual(fp1, fp2)

    def test_reindex_force_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "index", str(FIXTURES / "alpha.txt"))
            code, payload = _run(db, "reindex", "--force")
            self.assertEqual(code, 0)
            self.assertGreaterEqual(payload["totals"]["indexed"], 1)
            self.assertTrue(payload.get("generation_activated"))

    def test_host_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "allowed_hosts", "127.0.0.1,localhost")
            code, payload = _run(
                db,
                "config",
                "set-ollama",
                "--url",
                "https://evil.example.com",
                "--model",
                "x",
            )
            self.assertEqual(code, 1)
            self.assertEqual(payload["error"]["type"], "HostNotAllowed")

    def test_index_root_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            root = Path(tmp) / "allowed"
            root.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("hello", encoding="utf-8")
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "index_root", str(root))
            code, payload = _run(db, "index", str(outside))
            self.assertEqual(code, 1)
            self.assertEqual(payload["error"]["type"], "PathOutsideRoot")

    def test_context_untrusted_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "index", str(FIXTURES / "alpha.txt"))
            _, q = _run(db, "query", "data mesh")
            self.assertIn("UNTRUSTED_RETRIEVED_CONTENT", q["context"])
            self.assertTrue(q["meta"]["content_untrusted"])

    def test_max_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            big = Path(tmp) / "big.txt"
            big.write_text("x" * 5000, encoding="utf-8")
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "max_file_bytes", "100")
            code, payload = _run(db, "index", str(big))
            self.assertEqual(code, 1)
            self.assertEqual(payload["results"][0]["status"], "error")

    def test_schema_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            code, payload = _run(db, "schema", "query")
            self.assertEqual(code, 0)
            js = payload["json_schema"]
            self.assertEqual(js["command"], "query")
            self.assertIn("properties", js["args"])
            self.assertIn("text", js["args"]["properties"])

    def test_docs_delete_requires_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "index", str(FIXTURES / "alpha.txt"))
            code, payload = _run(db, "docs", "delete", "alpha")
            self.assertEqual(code, 1)
            self.assertEqual(payload["error"]["type"], "NotFound")

    def test_health_hash_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, payload = _run(db, "health")
            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "ready")
