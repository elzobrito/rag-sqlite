# rag-sqlite — Referência do CLI

Sintaxe, flags, settings, exit codes e schemas de saída do `rag_sqlite.py`.

Espírito alinhado à [Referência do CLI do ESAA-Core](../../../ESAA-Core/docs/guides/esaa-cli-reference.md):
referência densa; exemplos de fluxo ficam nos [cenários](rag-sqlite-cenarios.md).

## Invocação

```bash
python rag_sqlite.py [global options] <command> [command options] [args]
```

Equivale a executar o script na raiz do repositório. Não há pacote PyPI
obrigatório neste MVP.

### Opções globais

| Opção | Default | Descrição |
|-------|---------|-----------|
| `--db PATH` | `./rag.sqlite` ou `$RAG_SQLITE_DB` | Path do SQLite |
| `--compact` | off | JSON sem indent |
| `--verbose` | off | Logs em stderr |
| `--create` | off | Cria DB se ausente (mesmo em leituras) |
| `-h` / `--help` | — | Ajuda (uso humano; erros de uso em modo normal vão como JSON) |

## Criação do banco

| Comando | Cria DB se faltar? |
|---------|-------------------|
| `init` | sim |
| `config set`, `config set-ollama`, `config reset` | sim |
| `index` | sim |
| `query`, `stats`, `health`, `docs *`, `config list/get` | **não** (erro `DB_NOT_FOUND`) |
| qualquer + `--create` | sim |
| `schema` | não usa DB |

## Comandos

### `init`

Garante schema, seed de settings e meta.

```bash
python rag_sqlite.py --db ./kb.sqlite init
```

Schema saída: `rag_sqlite.init.v1` — `created`, `migrated`, `schema_db_version`.

### `config list`

Lista todas as settings tipadas.

```bash
python rag_sqlite.py --db ./kb.sqlite config list
```

### `config get KEY`

```bash
python rag_sqlite.py --db ./kb.sqlite config get hybrid_alpha
```

### `config set KEY VALUE`

```bash
python rag_sqlite.py --db ./kb.sqlite config set embedding_provider hash
python rag_sqlite.py --db ./kb.sqlite config set hybrid_alpha 0.7
```

Valida tipo e ranges; rejeita `NaN`/`Inf` em floats.

### `config reset [KEY]`

Restaura default de uma key ou de todas.

### `config set-ollama`

Atualização **atômica** de provider ollama + URL + model (+ timeout opcional).

```bash
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url http://127.0.0.1:11434 \
  --model embeddinggemma \
  --timeout 120
```

| Flag | Obrigatório | Descrição |
|------|-------------|-----------|
| `--url` | sim | Base URL (local ou remota) |
| `--model` | sim | Nome do modelo de embedding |
| `--timeout` | não | Segundos HTTP (`timeout_seconds`) |

Se qualquer valor for inválido, **nenhuma** key parcial é gravada.

### `index PATH`

Indexa arquivo ou diretório recursivo.

```bash
python rag_sqlite.py --db ./kb.sqlite index ./tests/fixtures
python rag_sqlite.py --db ./kb.sqlite index ./docs --force
python rag_sqlite.py --db ./kb.sqlite index ./docs --sync
python rag_sqlite.py --db ./kb.sqlite index ./docs --sync --prune
```

| Flag | Descrição |
|------|-----------|
| `--force` | Força rebuild / nova geração quando aplicável |
| `--sync` | Lista documentos no DB sob o path que sumiram do disco |
| `--prune` | Com `--sync`, remove os missing (explícito) |

Extensões: setting `index_extensions` (default `.txt,.md`).

Schema: `rag_sqlite.index.v1` — `totals`, `results[]`, `index_fingerprint`,
`generation_id`, `generation_activated`.

### `reindex`

Reprocessa todos os `documents` conhecidos.

```bash
python rag_sqlite.py --db ./kb.sqlite reindex
python rag_sqlite.py --db ./kb.sqlite reindex --force
```

### `docs list` | `docs show REF` | `docs delete REF`

```bash
python rag_sqlite.py --db ./kb.sqlite docs list
python rag_sqlite.py --db ./kb.sqlite docs show 1
python rag_sqlite.py --db ./kb.sqlite docs show /abs/path/file.txt
python rag_sqlite.py --db ./kb.sqlite docs delete 1
```

`delete` aceita **somente** id numérico ou path absoluto exato (não substring).

### `query TEXT`

```bash
python rag_sqlite.py --db ./kb.sqlite query "data mesh" \
  --top-k 5 \
  --min-score 0.1 \
  --min-score-relative 0.9 \
  --hybrid-alpha 0.7 \
  --expand-neighbors 1 \
  --doc alpha \
  --path /home/elzobrito/desenvolvimento/rag-sqlite/tests
```

| Flag | Default | Descrição |
|------|---------|-----------|
| `--top-k` | setting `top_k` | Máx. hits primários (cap `max_top_k`) |
| `--min-score` | setting | Piso absoluto do hybrid |
| `--min-score-relative` | null | Mantém hits ≥ fração × top1 |
| `--hybrid-alpha` | setting / 0 se hash | Peso do cosine |
| `--expand-neighbors` | setting | ±N chunks na mesma doc |
| `--doc` | — | id ou substring de filename |
| `--path` | — | filtro ILIKE em `source_path` |

Schema: `rag_sqlite.query.v1` — `meta`, `hits[]`, `context`.

### `export-context TEXT`

Mesmos filtros de `query`; saída reduzida.

Schema: `rag_sqlite.export_context.v1`.

### `stats`

Contagens, fingerprint atual, geração ativa, resumo de settings.

### `health`

```bash
python rag_sqlite.py --db ./kb.sqlite health
```

| `status` | Significado |
|----------|-------------|
| `ready` | Operacional |
| `degraded` | DB ok, provider com ressalva (modelo ausente, sem geração, enabled=false) |
| `unhealthy` | Falha de conexão ou config inválida; exit `1` |

### `schema [name]`

```bash
python rag_sqlite.py schema
python rag_sqlite.py schema query
```

Não exige DB. Devolve JSON Schema de argumentos/saídas para tool-calling.

## Settings (tabela `settings`)

| Key | Tipo | Default | Notas |
|-----|------|---------|-------|
| `enabled` | bool | true | Desliga embed |
| `embedding_provider` | str | ollama | `ollama` \| `hash` |
| `embedding_model` | str | embeddinggemma | Nome no Ollama |
| `base_url` | str | http://127.0.0.1:11434 | Sem path `/api/...` |
| `chunk_size_chars` | int | 1200 | ≥ 100 |
| `chunk_overlap_chars` | int | 200 | &lt; chunk_size |
| `batch_size` | int | 32 | Batch embed |
| `timeout_seconds` | int | 120 | HTTP |
| `hybrid_alpha` | float | 0.7 | \[0,1\] |
| `top_k` | int | 5 | Default query |
| `min_score` | float | 0.0 | Piso |
| `expand_neighbors` | int | 0 | Vizinhos |
| `score_decimals` | int | 6 | Arredondamento |
| `index_extensions` | str | .txt,.md | CSV |
| `allowed_hosts` | str | * | CSV hosts ou `*` |
| `index_root` | str | "" | Path absoluto se setado |
| `allow_symlinks` | bool | false | — |
| `max_file_bytes` | int | 2000000 | 0 = sem limite prático se 0? (implementação: 0 desliga check se falsy) |
| `max_top_k` | int | 50 | Cap de top_k |
| `context_max_chars` | int | 50000 | Truncate context |
| `max_chunks_per_doc` | int | 500 | Cap de chunks |
| `health_probe_embed` | bool | false | Probe `/api/embed` |

## Exit codes

| Code | Quando |
|------|--------|
| 0 | Sucesso (query com 0 hits permanece 0 se `ok`) |
| 1 | Erro operacional ou de uso |
| 2 | `index` sem arquivos candidatos (e ok estrutural) |

## Schemas de saída (prefixo)

| Schema | Comando |
|--------|---------|
| `rag_sqlite.error.v1` | qualquer falha |
| `rag_sqlite.init.v1` | init |
| `rag_sqlite.config.*.v1` | config * |
| `rag_sqlite.index.v1` | index |
| `rag_sqlite.reindex.v1` | reindex |
| `rag_sqlite.docs.*.v1` | docs * |
| `rag_sqlite.query.v1` | query |
| `rag_sqlite.export_context.v1` | export-context |
| `rag_sqlite.stats.v1` | stats |
| `rag_sqlite.health.v1` | health |
| `rag_sqlite.schema.v1` | schema |

## Score

```text
hybrid = α * cosine + (1 - α) * keyword
```

- cosine: similaridade de vetores (fail se dims divergem)
- keyword: |q ∩ d| / |q| após casefold e strip de acentos
- provider `hash` sem override: α = 0

## Ambiente

| Variável | Efeito |
|----------|--------|
| `RAG_SQLITE_DB` | Default de `--db` se a flag omitida |
