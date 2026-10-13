"""Testes unitarios e de integracao do micro-servidor scripts/hf_embed_server.py."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rag_sqlite as rag
from scripts.hf_embed_server import create_server, mock_embedder


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestHfEmbedServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.host = "127.0.0.1"
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.server = create_server(
            host=cls.host,
            port=cls.port,
            embed_fn=mock_embedder(768),
            model_name="google/embeddinggemma-2",
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_server_tags_endpoint(self):
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("models", data)
            model_names = [m["name"] for m in data["models"]]
            self.assertIn("google/embeddinggemma-2", model_names)
            self.assertIn("embeddinggemma-2", model_names)

    def test_server_embed_endpoint(self):
        url = f"{self.base_url}/api/embed"
        payload = json.dumps({
            "model": "google/embeddinggemma-2",
            "input": ["Primeiro texto", "Segundo texto"],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("embeddings", data)
            self.assertEqual(len(data["embeddings"]), 2)
            self.assertEqual(len(data["embeddings"][0]), 768)
            self.assertEqual(len(data["embeddings"][1]), 768)

    def test_rag_sqlite_integration_with_hf_embed_server(self):
        # 1. Valida tags via helper de rag_sqlite
        tags = rag.ollama_tags(self.base_url, timeout_seconds=5)
        self.assertIn("embeddinggemma-2", tags)

        # 2. Valida embed_ollama consumindo o hf_embed_server
        vectors = rag.embed_ollama(
            ["Data Mesh", "Database"],
            base_url=self.base_url,
            model="embeddinggemma-2",
            timeout_seconds=5,
        )
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 768)

        # 3. Valida ciclo completo via CLI rag_sqlite (config + index + health + query)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test_hf.sqlite"
            fixtures = Path(__file__).resolve().parent / "fixtures" / "alpha.txt"
            code_i, _ = _run(db, "init")
            self.assertEqual(code_i, 0)
            code_u, _ = _run(db, "config", "set", "base_url", self.base_url)
            self.assertEqual(code_u, 0)
            code_m, _ = _run(db, "config", "set", "embedding_model", "embeddinggemma-2")
            self.assertEqual(code_m, 0)
            code_idx, idx_res = _run(db, "index", str(fixtures))
            self.assertEqual(code_idx, 0)
            self.assertTrue(idx_res["ok"])

            code_h, health = _run(db, "health")
            self.assertEqual(code_h, 0)
            self.assertTrue(health["ok"])
            self.assertEqual(health["status"], "ready")
            self.assertTrue(health["provider_ok"])
            self.assertTrue(health["model_present"])

            code_q, q_res = _run(db, "query", "Data Mesh")
            self.assertEqual(code_q, 0)
            self.assertTrue(q_res["ok"])
            self.assertGreaterEqual(q_res["meta"]["hit_count"], 1)


def _run(db: Path, *args: str) -> tuple[int, dict]:
    cmd = [sys.executable, str(ROOT / "rag_sqlite.py"), "--db", str(db), "--compact", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, json.loads(proc.stdout)


if __name__ == "__main__":
    unittest.main()
