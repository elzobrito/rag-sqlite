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


class TestSqliteVec(unittest.TestCase):
    def test_backend_auto_prefers_sqlite_vec_when_available(self):
        if not rag.sqlite_vec_importable():
            self.skipTest("sqlite-vec not installed")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "vector_backend", "auto")
            _run(db, "index", str(FIXTURES))
            code, q = _run(db, "query", "data mesh", "--top-k", "2")
            self.assertEqual(code, 0)
            self.assertTrue(q["ok"])
            self.assertEqual(q["meta"]["backend"], "sqlite-vec")
            self.assertTrue(q["meta"]["sqlite_vec"]["available"])
            self.assertTrue(q["meta"]["sqlite_vec"]["knn"]["used"])
            self.assertGreaterEqual(q["meta"]["hit_count"], 1)
            self.assertEqual(q["hits"][0]["filename"], "alpha.txt")

    def test_backend_python_forces_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "vector_backend", "python")
            _run(db, "index", str(FIXTURES))
            code, q = _run(db, "query", "data mesh", "--top-k", "2")
            self.assertEqual(code, 0)
            self.assertEqual(q["meta"]["backend"], "python")
            self.assertFalse(q["meta"]["sqlite_vec"]["knn"]["used"])

    def test_reindex_force_rebuilds_vec(self):
        if not rag.sqlite_vec_importable():
            self.skipTest("sqlite-vec not installed")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "index", str(FIXTURES / "alpha.txt"))
            code, payload = _run(db, "reindex", "--force")
            self.assertEqual(code, 0)
            code_h, health = _run(db, "health")
            self.assertEqual(code_h, 0)
            self.assertTrue(health["sqlite_vec"]["available"])
            self.assertTrue(health["sqlite_vec"]["table_ready"])

    def test_strict_sqlite_vec_fails_closed_when_unavailable(self):
        # Unit: forced backend with vec_loaded=False always fails closed.
        with self.assertRaises(rag.CliError) as ctx:
            rag.resolve_vector_backend({"vector_backend": "sqlite-vec"}, False)
        self.assertEqual(ctx.exception.error_type, "SqliteVecUnavailable")
        if not rag.sqlite_vec_importable():
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "t.sqlite"
                _run(db, "config", "set", "embedding_provider", "hash")
                _run(db, "config", "set", "vector_backend", "sqlite-vec")
                _run(db, "index", str(FIXTURES / "alpha.txt"))
                code_q, q = _run(db, "query", "x")
                self.assertEqual(code_q, 1)
                self.assertEqual(q["error"]["type"], "SqliteVecUnavailable")

    def test_default_embedding_model_is_embeddinggemma_2(self):
        self.assertEqual(rag.DEFAULT_SETTINGS["embedding_model"]["value"], "embeddinggemma-2")
        self.assertEqual(rag.effective_model({"embedding_provider": "ollama"}), "embeddinggemma-2")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            code_init, init_res = _run(db, "init")
            self.assertEqual(code_init, 0)
            code, cfg = _run(db, "config", "get", "embedding_model")
            self.assertEqual(code, 0)
            self.assertEqual(cfg["value"], "embeddinggemma-2")

    def test_index_progress_feedback_in_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            # Executa sem --compact e sem --quiet para validar progresso no stderr
            cmd = [sys.executable, str(SCRIPT), "--db", str(db), "index", str(FIXTURES)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertIn("[index] Iniciando indexação", proc.stderr)
            self.assertIn("[index] Concluído:", proc.stderr)

    def test_index_quiet_suppresses_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            cmd = [sys.executable, str(SCRIPT), "--db", str(db), "index", str(FIXTURES), "--quiet"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertNotIn("[index]", proc.stderr)


_EMBEDDINGGEMMA_FINGERPRINT = "b913804a399f63ec6d50527d1652617e8ba9d3c1d8411c13bfe4e01534dd2c5d"


class TestExcludeDirsAndQueryDedup(unittest.TestCase):
    def test_parse_exclude_dirs(self):
        self.assertEqual(rag.parse_exclude_dirs(" build, ,.git,"), {"build", ".git"})
        self.assertEqual(rag.parse_exclude_dirs(""), set())

    def test_index_fingerprint_ignores_exclude_dirs_and_dedup_hits(self):
        self.assertEqual(rag.SCHEMA_VERSION, 2)
        base = {
            "embedding_provider": "ollama",
            "embedding_model": "embeddinggemma",
            "base_url": "http://127.0.0.1:11434",
            "chunk_size_chars": 1200,
            "chunk_overlap_chars": 200,
        }
        self.assertEqual(rag.index_fingerprint(base), _EMBEDDINGGEMMA_FINGERPRINT)
        richer = dict(
            base,
            exclude_dirs="build,.git,__pycache__",
            dedup_hits=True,
            dedup_max_jaccard=0.65,
            latest_run_only=False,
        )
        self.assertEqual(rag.index_fingerprint(richer), _EMBEDDINGGEMMA_FINGERPRINT)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "ollama")
            _run(db, "config", "set", "embedding_model", "embeddinggemma")
            _run(db, "config", "set", "base_url", "http://127.0.0.1:11434")
            _run(db, "config", "set", "chunk_size_chars", "1200")
            _run(db, "config", "set", "chunk_overlap_chars", "200")
            code, stats = _run(db, "stats")
            self.assertEqual(code, 0)
            self.assertEqual(stats["index_fingerprint"], _EMBEDDINGGEMMA_FINGERPRINT)
            self.assertEqual(stats["schema_db_version"], 2)
            self.assertEqual(
                stats["settings"]["exclude_dirs"],
                "build,.git,__pycache__,.venv,node_modules,.roadmap",
            )
            self.assertIs(stats["settings"]["dedup_hits"], True)

    def test_index_exclude_dirs_skips_build_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            (root / "materiais").mkdir(parents=True)
            (root / "build" / "materiais").mkdir(parents=True)
            body = "conteudo da aula sobre vetores e recuperacao"
            (root / "materiais" / "aula.md").write_text(body, encoding="utf-8")
            (root / "build" / "materiais" / "aula.md").write_text(body, encoding="utf-8")
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, indexed = _run(db, "index", str(root))
            self.assertEqual(code, 0)
            self.assertTrue(indexed["ok"])
            code_d, docs = _run(db, "docs", "list")
            self.assertEqual(code_d, 0)
            self.assertEqual(docs["count"], 1)
            rel = Path(docs["documents"][0]["source_path"]).relative_to(root.resolve())
            self.assertEqual(rel.as_posix(), "materiais/aula.md")

            explicit = Path(tmp) / "explicit.sqlite"
            _run(explicit, "config", "set", "embedding_provider", "hash")
            code_file, indexed_file = _run(
                explicit, "index", str(root / "build" / "materiais" / "aula.md")
            )
            self.assertEqual(code_file, 0)
            self.assertEqual(indexed_file["results"][0]["status"], "indexed")

            _run(db, "config", "set", "exclude_dirs", "")
            code2, indexed2 = _run(db, "index", str(root))
            self.assertEqual(code2, 0)
            self.assertTrue(indexed2["ok"])
            _, docs2 = _run(db, "docs", "list")
            rels = sorted(
                Path(doc["source_path"]).relative_to(root.resolve()).as_posix()
                for doc in docs2["documents"]
            )
            self.assertEqual(rels, ["build/materiais/aula.md", "materiais/aula.md"])

    def test_no_storage_dedup_by_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            (root / "a").mkdir(parents=True)
            (root / "b").mkdir(parents=True)
            body = "texto identico para proveniencia de content hash"
            (root / "a" / "doc.md").write_text(body, encoding="utf-8")
            (root / "b" / "doc.md").write_text(body, encoding="utf-8")
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "exclude_dirs", "")
            code, indexed = _run(db, "index", str(root))
            self.assertEqual(code, 0)
            self.assertTrue(indexed["ok"])
            _, docs = _run(db, "docs", "list")
            self.assertEqual(docs["count"], 2)
            self.assertEqual(len({doc["content_hash"] for doc in docs["documents"]}), 1)
            self.assertEqual(sum(int(doc["chunk_count"]) for doc in docs["documents"]), 2)
            a_path = str((root / "a").resolve())
            b_path = str((root / "b").resolve())
            _, qa = _run(db, "query", "proveniencia", "--path", a_path, "--top-k", "5")
            _, qb = _run(db, "query", "proveniencia", "--path", b_path, "--top-k", "5")
            self.assertTrue(qa["hits"])
            self.assertTrue(qb["hits"])
            self.assertTrue(all(h["source_path"].startswith(a_path + "/") for h in qa["hits"]))
            self.assertTrue(all(h["source_path"].startswith(b_path + "/") for h in qb["hits"]))
            self.assertNotEqual(qa["hits"][0]["document_id"], qb["hits"][0]["document_id"])

    def test_query_dedup_hits_and_no_dedup_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            (root / "a").mkdir(parents=True)
            (root / "b").mkdir(parents=True)
            body = "texto identico para deduplicacao de hits no top k"
            (root / "a" / "doc.md").write_text(body, encoding="utf-8")
            (root / "b" / "doc.md").write_text(body, encoding="utf-8")
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            _run(db, "config", "set", "exclude_dirs", "")
            code, indexed = _run(db, "index", str(root))
            self.assertEqual(code, 0)
            self.assertGreaterEqual(indexed["totals"]["indexed"], 2)
            code_q, deduped = _run(db, "query", "deduplicacao", "--top-k", "5")
            self.assertEqual(code_q, 0)
            self.assertTrue(deduped["ok"])
            self.assertIs(deduped["meta"]["dedup_hits"], True)
            self.assertEqual(len(deduped["hits"]), 1)
            self.assertEqual(len({h["text_hash"] for h in deduped["hits"]}), 1)
            code_n, both = _run(db, "query", "deduplicacao", "--top-k", "5", "--no-dedup")
            self.assertEqual(code_n, 0)
            self.assertIs(both["meta"]["dedup_hits"], False)
            self.assertEqual(len(both["hits"]), 2)
            self.assertEqual(len({h["text_hash"] for h in both["hits"]}), 1)
            self.assertEqual(len({h["source_path"] for h in both["hits"]}), 2)
            code_bad, bad = _run(db, "query", "deduplicacao", "--dedup", "--no-dedup")
            self.assertEqual(code_bad, 1)
            self.assertEqual(bad["schema_version"], "rag_sqlite.error.v1")
            self.assertEqual(json.loads(json.dumps(bad))["ok"], False)

    def test_token_jaccard_edges(self):
        self.assertEqual(rag.token_jaccard(set(), set()), 1.0)
        self.assertEqual(rag.token_jaccard(set(), {"a"}), 0.0)
        self.assertEqual(rag.token_jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_query_near_dedup_jaccard_collapses_variants(self):
        common = (
            "As cinco perspectivas de Garvin e custos da qualidade organizam a aula. "
            "A visao transcendental reconhece excelencia dificil de definir ao ver uma obra. "
            "A visao baseada no produto liga qualidade a caracteristicas inerentes, como capacidade e latencia de memoria. "
            "A visao da manufatura trata qualidade como conformidade com a especificacao e reduz desperdicio. "
            "A visao do usuario atende as metas de quem usa o produto. "
            "A visao baseada no valor compara beneficio percebido e preco pago."
        )
        variants = {
            "garvin_a.md": common + " Variante do roteiro inicial.",
            "garvin_b.md": common.replace("organizam a aula", "organizam o roteiro").replace("uma obra", "um quadro")
            + " Variante do material revisado.",
            "garvin_c.md": common.replace("organizam a aula", "organizam o encontro").replace(
                "latencia de memoria", "latencia do disco"
            )
            + " Variante da fala.",
        }
        costs = (
            "Custos da qualidade se dividem em prevencao, avaliacao e falha. "
            "Prevencao investe em treinamento, padroes e revisoes antes do defeito. "
            "Avaliacao inspeciona, testa e audita o produto. "
            "Falha interna aparece antes da entrega e falha externa gera retrabalho, garantia e insatisfacao do cliente."
        )
        garvin_sets = [set(rag.tokenize(text)) for text in variants.values()]
        cost_tokens = set(rag.tokenize(costs))
        for left, right in zip(garvin_sets, garvin_sets[1:]):
            self.assertGreater(rag.token_jaccard(left, right), 0.70)
        for tokens in garvin_sets:
            self.assertLess(rag.token_jaccard(tokens, cost_tokens), 0.65)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            for name, text in variants.items():
                (root / name).write_text(text, encoding="utf-8")
            (root / "custos.md").write_text(costs, encoding="utf-8")
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, indexed = _run(db, "index", str(root))
            self.assertEqual(code, 0)
            self.assertEqual(indexed["totals"]["indexed"], 4)
            _, docs = _run(db, "docs", "list")
            self.assertEqual(docs["count"], 4)
            query = "cinco perspectivas de Garvin e custos da qualidade"
            code_q, near = _run(db, "query", query, "--top-k", "2")
            self.assertEqual(code_q, 0)
            self.assertIs(near["meta"]["dedup_hits"], True)
            self.assertEqual(near["meta"]["dedup_max_jaccard"], 0.65)
            self.assertEqual(len(near["hits"]), 2)
            self.assertTrue(near["hits"][0]["filename"].startswith("garvin_"))
            self.assertEqual(near["hits"][1]["filename"], "custos.md")
            self.assertIn("prevencao", near["hits"][1]["chunk_text"])
            code_w, wide = _run(db, "query", query, "--top-k", "2", "--dedup-jaccard", "1.0")
            self.assertEqual(code_w, 0)
            self.assertEqual(wide["meta"]["dedup_max_jaccard"], 1.0)
            self.assertEqual(wide["hits"][0]["source_path"], near["hits"][0]["source_path"])
            self.assertTrue(all(h["filename"].startswith("garvin_") for h in wide["hits"]))
            code_n, raw = _run(db, "query", query, "--top-k", "2", "--no-dedup")
            self.assertEqual(code_n, 0)
            self.assertIs(raw["meta"]["dedup_hits"], False)
            self.assertTrue(all(h["filename"].startswith("garvin_") for h in raw["hits"]))

    def test_query_latest_run_only_filters_older_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            old = root / "aula-01" / "20260820T145302Z" / "materiais"
            new = root / "aula-01" / "20260820T162632Z" / "materiais"
            loose = root / "avulso"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            loose.mkdir(parents=True)
            (old / "nota.md").write_text(
                "marcadorcomum conteudo antigo da run anterior sobre alfa",
                encoding="utf-8",
            )
            (new / "nota.md").write_text(
                "marcadorcomum conteudo recente da run atual sobre beta",
                encoding="utf-8",
            )
            (loose / "doc.md").write_text(
                "marcadorcomum documento avulso sem timestamp sobre gamma",
                encoding="utf-8",
            )
            db = Path(tmp) / "t.sqlite"
            _run(db, "config", "set", "embedding_provider", "hash")
            code, indexed = _run(db, "index", str(root))
            self.assertEqual(code, 0)
            self.assertEqual(indexed["totals"]["indexed"], 3)
            _, docs = _run(db, "docs", "list")
            self.assertEqual(docs["count"], 3)
            code_q, filtered = _run(
                db, "query", "marcadorcomum", "--top-k", "5", "--latest-run", "--no-dedup"
            )
            self.assertEqual(code_q, 0)
            self.assertIs(filtered["meta"]["latest_run_only"], True)
            paths = [h["source_path"] for h in filtered["hits"]]
            self.assertEqual(len(paths), 2)
            self.assertTrue(any("20260820T162632Z" in p for p in paths))
            self.assertTrue(any(p.endswith("/avulso/doc.md") for p in paths))
            self.assertFalse(any("20260820T145302Z" in p for p in paths))
            _, everything = _run(db, "query", "marcadorcomum", "--top-k", "5", "--all-runs", "--no-dedup")
            self.assertIs(everything["meta"]["latest_run_only"], False)
            self.assertTrue(any("20260820T145302Z" in h["source_path"] for h in everything["hits"]))

