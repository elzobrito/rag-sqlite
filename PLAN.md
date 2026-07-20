# PLAN — RAG autossuficiente: um script Python + SQLite (LLM-ready)

**Status:** plano completo executado (RAG-001…RAG-009 done)  
**Data:** 2026-07-20  
**Origem:** TOP-011 / flask-dashboard `rag.py` + `TextEmbeddingService`  
**Entrega alvo:** ferramenta standalone, sem Flask/Postgres/RQ  

**Path:** `/home/elzobrito/desenvolvimento/rag-sqlite/PLAN.md`

## Revisão Codex (2026-07-20) — resumo e follow-up

**Veredito original:** conceito bom, MVP ainda não aprovável (contratos CLI, atomicidade, segurança, consistência de índice).

### Corrigido nesta rodada (evidência em testes)

| Pri | Problema Codex | Resolução |
|-----|----------------|-----------|
| P0 | argparse quebra stdout JSON | `JsonArgumentParser` → `rag_sqlite.error.v1` |
| P0 | `set-ollama` não atômico | `prepare_setting` + um único `commit` / rollback |
| P0 | query sem hits como “erro” (exit 2) | exit `0` + `ok:true` + `hit_count:0` |
| P1 | `NaN` em float settings / JSON | rejeição + `allow_nan=False` |
| P1 | leitura cria DB vazio | só `init`/`config set*`/`index`/`--create` |
| P1 | dim mismatch → cosine 0 silencioso | `DimensionMismatch` fail-closed |
| P1 | `hash` com α=0.7 semântica falsa | default α=0 (keyword) no provider hash |

### Fechado nesta execução (ESAA RAG-001…RAG-009)

- P0 index fingerprint + `reindex --force` (RAG-001)
- P0 gerações de índice building/active/abandoned (RAG-002)
- P0 schema v2 + migração v1→v2 + SchemaTooNew (RAG-003)
- P0 `allowed_hosts`, `index_root`, context untrusted (RAG-004)
- P1 limites + embedding float32 BLOB (RAG-005)
- P1 SAVEPOINT por arquivo + `--sync`/`--prune` (RAG-006)
- P1 health `ready|degraded|unhealthy` + probe opcional (RAG-007)
- P2 JSON Schema no comando `schema` (RAG-008)
- Governança ESAA bootstrapped em `.roadmap/` + closeout (RAG-009)

**Testes:** `python -m unittest tests.test_rag_sqlite -v` (28 casos offline).

---

## 1. Problema

O flask-dashboard já tem:

- ingestão de embeddings assíncrona (`page_text_embedding`, Ollama `embeddinggemma`)
- retrieval determinístico em `rag.py` (hybrid cosine + keyword)

Isso depende de app Flask, Postgres, workers e settings externos.  
Queremos o **mesmo ciclo cognitivo** em:

1. **um único script Python**
2. **um banco SQLite** (config + documentos + chunks + vetores)
3. **totalmente operável por comandos CLI** (adequado a LLM/tool-calling)
4. **JSON determinístico** em stdout

---

## 2. Objetivo

| Capacidade | Descrição |
|------------|-----------|
| Bootstrap | Se o SQLite não existir, o script **cria** arquivo + schema + seed |
| Config via comando | Ollama local **ou** servidor remoto, modelo de embedding, chunking, retrieval — **tudo no SQLite** |
| Index | Arquivo/pasta → chunk → embed → upsert |
| Query | Embed da pergunta → hybrid rank → `hits` + `context` |
| LLM-ready | Um JSON por execução, schemas versionados, `ok`, exit codes estáveis |

**Não é objetivo do MVP:** UI, API HTTP, geração de resposta pelo script, FAISS/pgvector, ESAA-Core.

---

## 3. Artefatos a criar

| Artefato | Path |
|----------|------|
| Script único | `desenvolvimento/rag-sqlite/rag_sqlite.py` |
| Este plano | `desenvolvimento/rag-sqlite/PLAN.md` |
| README (contrato LLM) | `desenvolvimento/rag-sqlite/README.md` |
| Testes | `desenvolvimento/rag-sqlite/tests/test_rag_sqlite.py` |
| Fixtures | `desenvolvimento/rag-sqlite/tests/fixtures/*.txt` |
| DB runtime (não versionar) | `./rag.sqlite` ou path via `--db` / `RAG_SQLITE_DB` |

**Dependências:** stdlib (`sqlite3`, `urllib`, `json`, `argparse`, `hashlib`, …).  
Provider offline `hash` embutido (testes sem Ollama). Ollama via HTTP.

---

## 4. Arquitetura

```text
python rag_sqlite.py [--db PATH] <command> ...

        │
        ▼
 ensure_db(path)
   se !exists → CREATE file + schema + seed settings
   se exists  → open + ensure tables + seed keys missing
        │
        ├─► settings (SQLite)     config list|get|set|reset|set-ollama
        ├─► documents / chunks    index | reindex | docs *
        ├─► embed provider        ollama (base_url local|remoto) | hash
        └─► retrieve hybrid       query → JSON + context

stdout: exatamente 1 objeto JSON
stderr: logs se --verbose
exit: 0 ok | 1 erro | 2 ok sem hits / nada a indexar
```

### Princípios para consumo por LLM

1. Stdout = JSON only (nunca misturar log)
2. Schemas `rag_sqlite.<cmd>.v1`
3. Campo `ok: true|false` em todo resultado
4. Retrieval determinístico (scores 6 casas; ordem estável)
5. Comando `schema` descreve a interface
6. Bootstrap automático: **não exige** `init` prévio
7. Overrides CLI (`--top-k` etc.) só na run; persistência só via `config *`

---

## 5. Auto-detecção e criação do SQLite

`ensure_db(db_path)` em **todo** comando que usa o banco:

1. Resolver path absoluto (`expanduser().resolve()`)
2. Criar diretório pai se faltar
3. Se arquivo **não** existe → create + schema + seed → `created=true`
4. Se existe → connect, `PRAGMA foreign_keys=ON`, garantir tables, `INSERT` de settings ausentes (nunca sobrescrever valores do usuário)
5. `init` opcional: só reporta `created` / `schema_ready` / contagem de seeds

**Aceite:** apagar o `.sqlite` e rodar `config set-ollama ...` recria o DB com as settings novas.

---

## 6. Schema SQLite

### 6.1 `meta`

```sql
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- schema_version = "1"
```

### 6.2 `settings` (configuração operacional)

```sql
CREATE TABLE settings (
  key         TEXT PRIMARY KEY,
  value       TEXT NOT NULL,
  value_type  TEXT NOT NULL,  -- bool|int|float|str
  description TEXT NOT NULL DEFAULT ''
);
```

### 6.3 Defaults seed

| key | default | papel |
|-----|---------|--------|
| `enabled` | `true` | liga/desliga embed |
| `embedding_provider` | `ollama` | `ollama` \| `hash` |
| `embedding_model` | `embeddinggemma` | modelo no servidor |
| `base_url` | `http://127.0.0.1:11434` | Ollama **local ou remoto** |
| `chunk_size_chars` | `1200` | janela |
| `chunk_overlap_chars` | `200` | overlap |
| `batch_size` | `32` | batch embed |
| `timeout_seconds` | `120` | HTTP (aumentar em remoto) |
| `hybrid_alpha` | `0.7` | peso cosine |
| `top_k` | `5` | default query |
| `min_score` | `0.0` | piso |
| `expand_neighbors` | `0` | vizinhos |
| `score_decimals` | `6` | arredondamento |
| `index_extensions` | `.txt,.md` | extensões ao indexar pasta |

**Local vs online = só `base_url` (e timeout/model):**

- Local: `http://127.0.0.1:11434`
- Remoto: `https://ollama.exemplo.com` ou `http://192.168.x.x:11434`

Normalizar: strip `/` final; se sem scheme, prefixar `http://`.  
Script anexa `/api/embed` e `/api/tags`.

### 6.4 `documents`

```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL UNIQUE,
  filename TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  char_count INTEGER NOT NULL,
  chunk_count INTEGER NOT NULL,
  mtime_ns INTEGER,
  indexed_at TEXT NOT NULL,
  status TEXT NOT NULL
);
```

### 6.5 `chunks`

```sql
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  embedding_json TEXT NOT NULL,
  UNIQUE(document_id, provider, model, content_hash, chunk_index)
);
CREATE INDEX idx_chunks_doc ON chunks(document_id);
CREATE INDEX idx_chunks_provider_model ON chunks(provider, model);
```

MVP usa JSON de floats (debugável). BLOB float32 = melhoria futura.

---

## 7. Superfície de comandos (contrato CLI)

```bash
python rag_sqlite.py [--db PATH] [--compact] [--verbose] <command> ...
```

### 7.1 Configuração (tudo comando → SQLite)

| Comando | Função | Schema |
|---------|--------|--------|
| `config list` | lista settings | `rag_sqlite.config.list.v1` |
| `config get KEY` | um valor tipado | `rag_sqlite.config.get.v1` |
| `config set KEY VALUE` | grava + valida | `rag_sqlite.config.set.v1` |
| `config reset [KEY]` | volta default(s) | `rag_sqlite.config.reset.v1` |
| **`config set-ollama`** | atalho atômico provider+url+model | `rag_sqlite.config.set_ollama.v1` |

```bash
# Local
python rag_sqlite.py config set-ollama \
  --url http://127.0.0.1:11434 \
  --model embeddinggemma

# Servidor online
python rag_sqlite.py config set-ollama \
  --url https://ollama.exemplo.com \
  --model nomic-embed-text \
  --timeout 180

# Equivalente granular
python rag_sqlite.py config set embedding_provider ollama
python rag_sqlite.py config set base_url https://ollama.exemplo.com
python rag_sqlite.py config set embedding_model embeddinggemma
```

Validações: key desconhecida → erro; provider ∉ {ollama,hash} → erro; ranges de chunk/alpha.

### 7.2 Dados e retrieval

| Comando | Função |
|---------|--------|
| `init` | ensure + relatório (opcional) |
| `index PATH` | indexa arquivo ou diretório recursivo |
| `reindex` | reprocessa docs com provider/model **atuais** |
| `docs list` / `show` / `delete` | inventário |
| `query "texto"` | hybrid retrieval + `context` |
| `stats` | contagens + resumo de config efetiva |
| `health` | DB ok; se ollama: ping `base_url` + tags/modelo |
| `schema [cmd]` | descoberta da interface para LLM |
| `export-context "q"` | JSON mínimo com `context` |

Flags de `query` (default do SQLite):  
`--top-k`, `--min-score`, `--min-score-relative`, `--hybrid-alpha`, `--expand-neighbors`, `--doc`, `--path`

### 7.3 Envelopes JSON

**Erro:**

```json
{
  "schema_version": "rag_sqlite.error.v1",
  "ok": false,
  "error": { "type": "ValueError", "message": "..." },
  "command": "query"
}
```

**Query sucesso:**

```json
{
  "schema_version": "rag_sqlite.query.v1",
  "ok": true,
  "query": "...",
  "meta": {
    "provider": "ollama",
    "model": "embeddinggemma",
    "base_url": "http://127.0.0.1:11434",
    "dimensions": 768,
    "top_k": 5,
    "hybrid_alpha": 0.7,
    "score_metric": "hybrid_cosine_keyword",
    "candidate_count": 42,
    "hit_count": 5,
    "deterministic": true,
    "db": "/abs/path/rag.sqlite"
  },
  "hits": [],
  "context": "[doc=1 file=x.txt chunk=0 score=0.81]\n..."
}
```

---

## 8. Fluxos internos

### 8.1 Index

1. `ensure_db`
2. Ler settings (provider, model, chunk_*, batch, base_url, timeout)
3. Expandir paths; extensões de `index_extensions`
4. `normalize_text` + `chunk_text` (fronteiras `\n\n` / `\n` / espaço — parity dashboard)
5. `content_hash = sha256(normalized)`
6. Se hashes idênticos ao existente (mesmo provider/model) → `unchanged`
7. Senão: `embed_batch` → transaction delete+insert chunks → update `documents`
8. JSON agregado por arquivo: `indexed|unchanged|empty|error`

### 8.2 Query

1. `ensure_db`
2. Embed da pergunta com provider/model do SQLite
3. `SELECT` chunks desse par (+ filtros doc/path)
4. Score: `hybrid = α*cosine + (1-α)*keyword` (token overlap normalizado)
5. Ordem: `-hybrid, -cosine, document_id, chunk_index, id`
6. `expand_neighbors` por `(document_id, chunk_index)`
7. Montar `hits` + `context`

### 8.3 Provider `ollama`

- `POST {base_url}/api/embed` `{"model","input"}`
- Health: `GET {base_url}/api/tags`
- Idêntico para localhost ou host remoto

### 8.4 Provider `hash` (offline)

Vetor estável derivado de SHA-256 (ex. 32 dims). Sem semântica. Testes e demo sem rede.

### 8.5 Troca de servidor/modelo

Chunks guardam `provider`/`model` da indexação.  
Query só usa o par **atual**.  
Se mismatch → 0 hits possível; `stats`/`health` alertam; mitigação: `reindex`.

---

## 9. Determinismo (checklist)

Portar de `flask-dashboard/rag.py` e `tests/test_rag_cli.py`:

- [ ] `round_score` 6 decimais
- [ ] cosine pure Python
- [ ] keyword: casefold + strip accents + tokens `[a-z0-9]+`
- [ ] hybrid com α ∈ [0,1]
- [ ] sort estável documentado
- [ ] neighbor expansion com deltas ordenados
- [ ] provider `hash` → mesma query + mesmo DB = mesmo ranking

---

## 10. Estrutura interna de `rag_sqlite.py`

Seções no mesmo arquivo (~500–800 linhas):

1. Constantes + `DEFAULT_SETTINGS`
2. JSON print, error envelope, exit codes
3. Text normalize / chunk / sha256
4. Scoring (cosine, keyword, hybrid, retrieve, expand)
5. Embed providers (ollama, hash)
6. SQLite layer (`ensure_db`, settings CRUD, docs/chunks)
7. Command handlers
8. `argparse` + `main()`

---

## 11. Playbook para LLM / operador

```bash
# Qualquer comando cria o DB se faltar
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url https://ollama.exemplo.com \
  --model embeddinggemma \
  --timeout 180

python rag_sqlite.py --db ./kb.sqlite config list
python rag_sqlite.py --db ./kb.sqlite health
python rag_sqlite.py --db ./kb.sqlite index ./corpus/
python rag_sqlite.py --db ./kb.sqlite query "pergunta do usuário" --top-k 5
# consumir campo .context
```

Offline:

```bash
python rag_sqlite.py config set embedding_provider hash
python rag_sqlite.py index ./corpus/
python rag_sqlite.py query "teste"
```

---

## 12. Testes obrigatórios

Arquivo: `tests/test_rag_sqlite.py` (preferir stdlib `unittest` se zero deps; pytest ok se já disponível).

| Caso | Esperado |
|------|----------|
| DB ausente + `config set` | cria arquivo; valor persiste |
| `config set-ollama` | grava provider, url, model atomicamente |
| `config get` tipagem | bool/int/float/str |
| key desconhecida | `error.v1`, exit 1 |
| segunda abertura | `created=false`, dados intactos |
| chunk + hybrid | scores/ordem estáveis |
| index + re-index unchanged | idempotência por hash |
| query provider hash | hits estáveis; exit 2 se vazio |
| expand_neighbors | inclui ±N |
| reindex após troca model | chunks refletem model novo |

Sem rede em CI. Health Ollama: mock ou skip.

---

## 13. Fases de implementação

| Fase | Conteúdo | Critério de done |
|------|----------|------------------|
| **1** | `ensure_db`, schema, `config *` incl. `set-ollama` | auto-create + persistência |
| **2** | chunk, provider `hash`, index/docs/query/stats | E2E offline |
| **3** | provider ollama, health, README local/remoto | E2E com Ollama se disponível |
| **4** | `schema`, `export-context`, exemplos tool-def | polish LLM |

**MVP entregável = Fases 1–3.**

---

## 14. Critérios de aceite (para o Codex validar)

1. [ ] Um script, zero Flask/Postgres/Redis.
2. [ ] Apagar SQLite + `config set-ollama` recria DB com settings corretas.
3. [ ] Configurar Ollama **local** e **URL remota** só com comandos; `config list` reflete.
4. [ ] `embedding_model` configurável por comando e usado em index/query.
5. [ ] Sem `.env` obrigatório (apenas `--db` / `RAG_SQLITE_DB` opcional).
6. [ ] `index` pasta de texto + `query` devolve `ok` + `context`.
7. [ ] Provider `hash`: mesma query → ranking idêntico (determinismo).
8. [ ] Testes offline passam.
9. [ ] Com Ollama alcançável: `health` ok + index/query funcionam.
10. [ ] Stdout sempre 1 JSON; erros em `rag_sqlite.error.v1`.

---

## 15. Riscos e mitigações

| Risco | Mitigação |
|-------|-----------|
| Servidor remoto lento/offline | `timeout_seconds` configurável; `health` claro |
| URL malformada | normalizar + validar em `set-ollama` |
| Index model A / query model B | alerta em stats/health; `reindex` |
| Auto-create em path errado | path absoluto em toda resposta config/init |
| Corpus grande / rank O(n) | filtro `--doc`; documentar limite; futuro índice |
| HTTPS self-signed | documentar limitação stdlib; extensão futura |

---

## 16. Decisões fixas

1. Standalone em `desenvolvimento/rag-sqlite/` — **não** embutir no flask-dashboard.
2. Auto-create SQLite no primeiro uso; `init` opcional.
3. Config operacional **só** no SQLite; mutação **só** por comandos.
4. Ollama local e remoto = mesmo provider; diferem `base_url` (+ model/timeout).
5. Atalho `config set-ollama` + `config set` genérico.
6. Providers MVP: `ollama` + `hash`.
7. Sem geração LLM no path default — só retrieval + `context`.
8. Stdout JSON only.

---

## 17. Referências no monorepo

| Peça | Path |
|------|------|
| Retrieval CLI atual | `desenvolvimento/flask-dashboard/rag.py` |
| Ingestão/chunk/embed | `desenvolvimento/flask-dashboard/services/text_embedding_service.py` |
| Modelo `PageTextEmbedding` | `desenvolvimento/flask-dashboard/models.py` |
| Seeds rag.* | `desenvolvimento/flask-dashboard/config_seeds.py` |
| Testes retrieval | `desenvolvimento/flask-dashboard/tests/test_rag_cli.py` |
| Tópico conversacional | TOP-011 (conversation-esaa) |

---

## 18. Escopo de avaliação para o Codex

Ao implementar ou revisar este plano, o Codex deve:

1. Confirmar que o desenho é implementável em **um** `.py` + SQLite stdlib.
2. Sinalizar gaps (ex.: auth em Ollama remoto, API key, TLS).
3. Implementar Fases 1–3 sem acoplar ao flask-dashboard.
4. Garantir testes offline e contrato JSON estável.
5. Não introduzir dependências pesadas sem necessidade (evitar sentence-transformers no MVP).

### Fora de escopo nesta avaliação

- Migrar dados do Postgres do flask-dashboard
- Substituir `rag.py` de produção
- Governança ESAA (claim/complete) para esta ferramenta utilitária

---

## 19. Ordem de implementação sugerida (checklist)

```text
[ ] Scaffold rag_sqlite.py + argparse + JSON helpers
[ ] ensure_db + schema + DEFAULT_SETTINGS seed
[ ] config list|get|set|reset + set-ollama
[ ] testes auto-create + config round-trip
[ ] chunk/normalize + provider hash
[ ] index + docs + idempotência
[ ] retrieve hybrid (port rag.py) + query + context
[ ] stats + export-context
[ ] provider ollama + health
[ ] reindex + mismatch warning
[ ] testes ranking/neighbors/exit codes
[ ] README playbook local + remoto + offline
```

---

## 20. Resultado esperado após implementação

```bash
$ rm -f ./kb.sqlite
$ python rag_sqlite.py --db ./kb.sqlite config set-ollama \
    --url http://127.0.0.1:11434 --model embeddinggemma
# → ok:true, db created, settings persisted

$ python rag_sqlite.py --db ./kb.sqlite index ./tests/fixtures
$ python rag_sqlite.py --db ./kb.sqlite query "texto da fixture" --top-k 3
# → hits + context determinísticos (ou semânticos se ollama)
```

Um LLM opera a base inteira só com essa CLI e o campo `context`.

---

## Nota operacional (Grok / plan mode)

- Plano canônico da sessão: este arquivo.  
- Pasta já criada: `/home/elzobrito/desenvolvimento/rag-sqlite/`  
- **Próximo passo humano:** aprovar para copiar este PLAN.md para o path de trabalho e/ou iniciar implementação.  
- Instrução ao Codex: avaliar seções 14–19; implementar conforme checklist §19 sem expandir escopo §2/§18.
