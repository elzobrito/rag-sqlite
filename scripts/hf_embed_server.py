"""Micro-servidor HTTP local que emula os endpoints /api/embed e /api/tags do Ollama,
permitindo servir modelos de embedding da Hugging Face (ex.: google/embeddinggemma-2).

Sem necessidade de dependencias web pesadas: utiliza a biblioteca padrao http.server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hf_embed_server")

EmbedCallable = Callable[[Sequence[str]], List[List[float]]]


def mock_embedder(dimensions: int = 768) -> EmbedCallable:
    """Gera vetores deterministas e normalizados (L2) para testes e execucao sem download."""
    def _embed(texts: Sequence[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            vec: List[float] = []
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            for i in range(dimensions):
                byte_val = seed[i % len(seed)]
                weight = ((byte_val ^ (i & 0xFF)) - 128) / 128.0
                vec.append(weight)
            # Normalizacao L2
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([round(x / norm, 7) for x in vec])
        return results

    return _embed


def load_hf_embedder(model_name: str, device: str = "auto") -> EmbedCallable:
    """Carrega o modelo do Hugging Face usando sentence-transformers ou transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Carregando '%s' via sentence-transformers...", model_name)
        dev = None if device == "auto" else device
        st_model = SentenceTransformer(model_name, device=dev)

        def _embed_st(texts: Sequence[str]) -> List[List[float]]:
            embeddings = st_model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in embeddings]

        return _embed_st

    except ImportError:
        pass

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        logger.info("Carregando '%s' via transformers/PyTorch...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_device = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        hf_model = AutoModel.from_pretrained(model_name).to(torch_device)
        hf_model.eval()

        def _embed_hf(texts: Sequence[str]) -> List[List[float]]:
            inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                outputs = hf_model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                # L2 norm
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                return [[float(x) for x in row] for row in normed.cpu().numpy()]

        return _embed_hf

    except ImportError:
        raise RuntimeError(
            "Nao foi possivel carregar o modelo localmente. "
            "Instale sentence-transformers executando: pip install sentence-transformers"
        )


class OllamaEmulationHandler(BaseHTTPRequestHandler):
    """Handler HTTP compativel com o contrato de API do Ollama."""

    embed_fn: EmbedCallable = staticmethod(mock_embedder(768))
    model_name: str = "google/embeddinggemma-2"

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/api/tags"):
            models_payload = {
                "models": [
                    {
                        "name": self.model_name,
                        "model": self.model_name,
                        "details": {"format": "pytorch", "family": "gemma4"},
                    },
                    {
                        "name": "embeddinggemma-2",
                        "model": "embeddinggemma-2",
                        "details": {"format": "pytorch", "family": "gemma4"},
                    },
                ]
            }
            body = json.dumps(models_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Endpoint nao encontrado")

    def do_POST(self) -> None:
        if self.path == "/api/embed":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                data = json.loads(raw_body.decode("utf-8"))

                input_val = data.get("input")
                if isinstance(input_val, str):
                    texts = [input_val]
                elif isinstance(input_val, list):
                    texts = [str(x) for x in input_val]
                else:
                    self.send_error(400, "Campo 'input' invalido (deve ser str ou list[str])")
                    return

                vectors = self.embed_fn(texts)

                resp = {
                    "model": data.get("model", self.model_name),
                    "embeddings": vectors,
                }
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                logger.exception("Erro ao processar /api/embed: %s", exc)
                err_body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
        else:
            self.send_error(404, "Endpoint POST nao encontrado")

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 11435,
    embed_fn: Optional[EmbedCallable] = None,
    model_name: str = "google/embeddinggemma-2",
) -> ThreadingHTTPServer:
    """Cria e retorna o servidor HTTP configurado."""
    handler_class = OllamaEmulationHandler
    if embed_fn is not None:
        handler_class.embed_fn = staticmethod(embed_fn)
    handler_class.model_name = model_name

    server = ThreadingHTTPServer((host, port), handler_class)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-servidor de embeddings compativel com Ollama")
    parser.add_argument("--host", default="127.0.0.1", help="Host para bind (padrao: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11435, help="Porta HTTP (padrao: 11435)")
    parser.add_argument("--model", default="google/embeddinggemma-2", help="Modelo Hugging Face (padrao: google/embeddinggemma-2)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo de computacao")
    parser.add_argument("--mock", action="store_true", help="Executa em modo mock (sem baixar pesos do modelo)")
    args = parser.parse_args()

    if args.mock:
        logger.info("Iniciando em modo MOCK (vetores 768d sinteticos deterministas)...")
        embed_fn = mock_embedder(768)
    else:
        embed_fn = load_hf_embedder(args.model, device=args.device)

    server = create_server(host=args.host, port=args.port, embed_fn=embed_fn, model_name=args.model)
    logger.info("Servidor ativo em http://%s:%d/ (Modelo: %s)", args.host, args.port, args.model)
    logger.info("Endpoints prontos: GET /api/tags, POST /api/embed")
    logger.info("Pressione Ctrl+C para encerrar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Encerrando servidor...")
        server.server_close()


if __name__ == "__main__":
    main()
