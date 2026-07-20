# rag-sqlite — RAG determinístico em um script + SQLite

> Um único CLI Python que indexa texto, gera embeddings e devolve contexto
> JSON estável para consumo por LLM — sem Flask, Postgres, Redis ou worker.

`rag-sqlite` aplica o mesmo ciclo cognitivo de um pipeline RAG (chunk → embed →
store → retrieve), com **fonte de verdade local em SQLite** e **saída
sempre JSON** em stdout, adequada a tool-calling de agentes.

Em uma linha:

```text
docs → index (chunk + embed) → SQLite → query (hybrid rank) → context para o LLM
```

Este repositório contém o runtime (`rag_sqlite.py`), fixtures, testes offline e
a documentação operacional. A execução de mudanças no produto, quando houver,
pode ser governada por **ESAA-Core** em `.roadmap/` (ver seção
[Governança ESAA](#governança-esaa)).

Plano de implementação e histórico de aceite: [`PLAN.md`](./PLAN.md).

## At A Glance

- **Um script**, stdlib Python 3.10+ (HTTP só para Ollama).
- **Config no SQLite** — mutável só por comandos (`config set`, `config set-ollama`).
- **DB auto-criado** em writes (`init`, `config set*`, `index`) ou com `--create`.
- **Stdout = um JSON** por invocação, inclusive erros de uso (`argparse`).
- **Retrieval determinístico** com score híbrido cosine + keyword.
- **Fingerprint + gerações** de índice (troca de modelo/URL exige `reindex --force`).
- **Providers:** `hash` (offline/teste, lexical) e `ollama` (semântico, local ou remoto).
- **LLM-ready:** campo `context` marcado como conteúdo não confiável + comando `schema`.

## Quickstart

Clone ou abra o diretório e use o interpretador Python do sistema (sem
dependências pip obrigatórias):

```bash
cd /home/elzobrito/desenvolvimento/rag-sqlite
python rag_sqlite.py --version 2>/dev/null || python rag_sqlite.py schema | head -c 200
```

### Offline (sem Ollama)

```bash
python rag_sqlite.py --db ./kb.sqlite config set embedding_provider hash
python rag_sqlite.py --db ./kb.sqlite index ./tests/fixtures
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --top-k 3 --min-score 0.1
```

Sinal esperado: `ok: true`, `hit_count >= 1`, `filename` do primeiro hit
`alpha.txt`, e um bloco `context` pronto para colar no prompt.

### Ollama local (semântico)

Pré-requisito: daemon Ollama em `127.0.0.1:11434` com o modelo de embedding
puxado (ex.: `ollama pull embeddinggemma`).

```bash
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url http://127.0.0.1:11434 \
  --model embeddinggemma

python rag_sqlite.py --db ./kb.sqlite health
python rag_sqlite.py --db ./kb.sqlite reindex --force
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --min-score 0.4
```

Após trocar provider/modelo/URL, o **fingerprint** muda: use `reindex --force`
(ou `index` de novo) antes de confiar na query.

### Ollama remoto

```bash
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url https://ollama.exemplo.com \
  --model embeddinggemma \
  --timeout 180

python rag_sqlite.py --db ./kb.sqlite config list
python rag_sqlite.py --db ./kb.sqlite health
```

Flags globais úteis:

| Flag | Função |
|------|--------|
| `--db PATH` | Arquivo SQLite (default: `./rag.sqlite` ou `$RAG_SQLITE_DB`) |
| `--compact` | JSON em uma linha |
| `--verbose` | Logs em stderr (stdout permanece só JSON) |
| `--create` | Cria o DB mesmo em comandos de leitura |

## Usage Guides

Documentação operacional (estilo cookbook / referência, no espírito dos guides
do [ESAA-Core](https://github.com/)):

| Guia | Conteúdo |
|------|----------|
| [Primeiros passos](docs/guides/rag-sqlite-getting-started.md) | Do zero ao primeiro `index` → `query` → `context` |
| [Cenários práticos (cookbook)](docs/guides/rag-sqlite-cenarios.md) | Situações reais: offline, Ollama, filtros, sync/prune, segurança, agente LLM |
| [Referência do CLI](docs/guides/rag-sqlite-cli-reference.md) | Subcomandos, flags, settings, exit codes e schemas de saída |
| [Agentes e tool-calling](docs/guides/rag-sqlite-llm-agents.md) | Contrato JSON, playbook para Grok/Codex/Claude, `schema` e `export-context` |

Plano e aceite histórico: [`PLAN.md`](./PLAN.md).

## When To Use rag-sqlite

Use quando precisar de:

- base de conhecimento **local** para um agente (docs `.txt`/`.md`)
- retrieval **reproduzível** e auditável (scores arredondados, ordem estável)
- configuração **persistida no próprio DB**, sem `.env` obrigatório
- Ollama local **ou** servidor remoto, trocável por comando
- zero stack pesada (sem vetor DB dedicado no MVP)

É provavelmente demais se você só precisa de uma busca `grep`, ou se o corpus
exige índice vetorial de milhões de chunks (pgvector/FAISS) — o rank atual é
O(n) em Python sobre o corpus candidatado.

## Why It Exists

Pipelines RAG “de demo” misturam chat, ingestão e storage em serviços
acoplados. Para um **agente de engenharia**, o que importa é:

| Problema | Resposta do rag-sqlite |
|----------|------------------------|
| Saída livre quebra tool-calling | Um objeto JSON versionado por comando |
| Config espalhada em env | Tabela `settings` no SQLite + CLI |
| Troca de modelo sem reindex silencioso | `index_fingerprint` + gerações ativas |
| Corpus parcial após falha | SAVEPOINT por arquivo; geração só ativa se ok |
| Prompt injection via trechos | Header `UNTRUSTED_RETRIEVED_CONTENT` no `context` |

## Architecture

```text
                    ┌─────────────────────────────┐
  CLI argv ───────► │        rag_sqlite.py        │
                    │  parse → JSON on error too  │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   settings (SQLite)      documents / chunks       embed provider
   config *               index / reindex          ollama | hash
                                  │
                                  ▼
                    index_fingerprint + generation
                                  │
                                  ▼
                         hybrid retrieve
                    cosine + keyword → context
```

| Camada | Responsabilidade |
|--------|------------------|
| CLI | Argumentos, envelopes JSON, exit codes |
| Settings | Defaults e overrides persistidos |
| Index | Chunk determinístico, embed em batch, BLOB float32 |
| Generation | `building` → `active` / `abandoned` por fingerprint |
| Query | Rank híbrido sobre a geração ativa do fingerprint atual |
| Health | `ready` \| `degraded` \| `unhealthy` |

## Core Concepts

**SQLite como store.** Um arquivo (`--db`) guarda `settings`, `documents`,
`chunks`, `index_generations` e `meta` (schema_version).

**Fingerprint.** Hash estável de provider + model + base_url (se ollama) +
parâmetros de chunk + versão de normalização. Query só usa chunks do fingerprint
**corrente**.

**Generation.** `reindex --force` (ou primeiro index de um fingerprint) abre uma
geração `building` e só a promove a `active` se o lote for utilizável. Query não
mistura gerações.

**Hybrid score.**

```text
hybrid = α * cosine + (1-α) * keyword
```

- Provider `hash`: α default **0** (só lexical — o vetor hash não é semântico).
- Provider `ollama`: α default **0.7** (settings `hybrid_alpha`).
- Scores com 6 casas; ordem: hybrid ↓, cosine ↓, document_id ↑, chunk_index ↑, id ↑.

**Fail-closed.** Path fora de `index_root`, host fora de `allowed_hosts`,
dimensões incompatíveis, `NaN` em settings e DB inexistente em leitura
produzem `rag_sqlite.error.v1` — não “sucesso vazio” enganoso.

## Repository Layout

```text
rag-sqlite/
  README.md                 este documento (onboarding público)
  PLAN.md                   plano e histórico de aceite
  AGENTS.md                 contrato operacional ESAA do workspace
  rag_sqlite.py             runtime CLI (artefato principal)
  kb.sqlite                 exemplo local de DB (não versionar se tiver dados reais)
  tests/
    fixtures/               corpus de demo (alpha.txt, beta.txt)
    test_rag_sqlite.py      testes offline (provider hash)
  docs/
    guides/                 guias de uso (getting started, cenários, CLI, agentes)
  .roadmap/                 ESAA-Core: event store e projeções (se governado)
```

Arquivos runtime que **não** devem ir para repositório público com dados reais:
`*.sqlite`, `.roadmap/activity.jsonl` se contiver contexto sensível.

## CLI Surface (resumo)

```bash
python rag_sqlite.py [--db PATH] [--compact] [--verbose] [--create] <command> ...
```

| Comando | Função |
|---------|--------|
| `init` | Garante schema + seed de settings |
| `config list\|get\|set\|reset` | Configuração no SQLite |
| `config set-ollama` | Provider ollama + URL + model (+ timeout), atômico |
| `index PATH` | Indexa arquivo/pasta (`.txt`/`.md`); `--force`, `--sync`, `--prune` |
| `reindex` | Reprocessa docs conhecidos; `--force` nova geração |
| `docs list\|show\|delete` | Inventário (`delete` exige id ou path **exato**) |
| `query TEXT` | Retrieval + `hits` + `context` |
| `export-context TEXT` | JSON enxuto focado em `context` |
| `stats` | Contagens e fingerprint ativo |
| `health` | DB + provider (`ready`/`degraded`/`unhealthy`) |
| `schema [cmd]` | JSON Schema / descoberta para agentes |

Sintaxe completa e exemplos: [Referência do CLI](docs/guides/rag-sqlite-cli-reference.md).

### Exit codes

| Código | Significado |
|--------|-------------|
| `0` | Sucesso (inclui query com zero hits se `ok: true`) |
| `1` | Erro (config, rede, path, uso, health unhealthy) |
| `2` | Index sem arquivos candidatos |

### Envelope de erro

```json
{
  "schema_version": "rag_sqlite.error.v1",
  "ok": false,
  "error": { "type": "UsageError", "message": "..." },
  "command": "query"
}
```

## Settings (principais)

Tudo via `config set KEY VALUE` ou `config set-ollama`.

| Key | Default | Papel |
|-----|---------|--------|
| `embedding_provider` | `ollama` | `ollama` \| `hash` |
| `embedding_model` | `embeddinggemma` | Modelo no servidor |
| `base_url` | `http://127.0.0.1:11434` | Ollama local ou remoto |
| `chunk_size_chars` | `1200` | Janela de chunk |
| `chunk_overlap_chars` | `200` | Overlap |
| `hybrid_alpha` | `0.7` | Peso do cosine |
| `top_k` / `max_top_k` | `5` / `50` | Hits e teto |
| `min_score` | `0.0` | Piso absoluto |
| `allowed_hosts` | `*` | Allowlist de hosts Ollama |
| `index_root` | `""` | Se setado, só indexa sob esse path |
| `max_file_bytes` | `2000000` | Tamanho máximo por arquivo |
| `context_max_chars` | `50000` | Truncamento do `context` |
| `health_probe_embed` | `false` | Probe opcional de `/api/embed` |

Lista completa: `python rag_sqlite.py --db ./kb.sqlite config list`.

## Security Notes

- `allowed_hosts` — restrija em produção (ex.: `127.0.0.1,localhost`).
- `index_root` — limite o que o agente pode indexar.
- `allow_symlinks` default `false`.
- Trechos no `context` são **dados não confiáveis**; o header
  `UNTRUSTED_RETRIEVED_CONTENT` instrui o LLM a não obedecer instruções neles.
- Não coloque API keys no SQLite neste MVP (provider OpenAI não é default).

## Tests

```bash
cd /home/elzobrito/desenvolvimento/rag-sqlite
python -m unittest tests.test_rag_sqlite -v
```

Os testes usam provider `hash` e **não** exigem Ollama. Sinal esperado: suite
verde (28 casos na linha de aceite do plano).

## Governança ESAA

Este workspace pode operar sob [ESAA-Core](../ESAA-Core/readme.md):

```bash
# No workspace rag-sqlite (já pode ter .roadmap/ do bootstrap)
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite verify
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite eligible
```

Guias oficiais do ESAA (padrões de CLI, cenários e runners) que inspiram a
documentação deste projeto:

- [ESAA — Primeiros passos](../ESAA-Core/docs/guides/esaa-getting-started.md)
- [ESAA — Cenários práticos](../ESAA-Core/docs/guides/esaa-cenarios.md)
- [ESAA — Referência do CLI](../ESAA-Core/docs/guides/esaa-cli-reference.md)
- [ESAA — Runners Codex/Claude](../ESAA-Core/docs/guides/esaa-runners-codex-claude-code.md)

Contrato local: [`AGENTS.md`](./AGENTS.md).

Regra: **não edite** `.roadmap/activity.jsonl` nem projeções à mão; use a CLI
`esaa` para claim/complete/review quando houver tarefas de produto.

## Minimal Agent Playbook

```bash
# 1) Descobrir a interface
python rag_sqlite.py schema
python rag_sqlite.py schema query

# 2) Garantir índice
python rag_sqlite.py --db ./kb.sqlite stats
python rag_sqlite.py --db ./kb.sqlite index ./docs

# 3) Recuperar contexto
python rag_sqlite.py --db ./kb.sqlite --compact export-context "pergunta do usuário" --top-k 5

# 4) Colar .context no system/user prompt do LLM
```

Detalhes e armadilhas: [Agentes e tool-calling](docs/guides/rag-sqlite-llm-agents.md).

## Status

| Item | Estado |
|------|--------|
| Runtime CLI | Operacional (`rag_sqlite.py`) |
| Testes offline | Suite unittest |
| Plano RAG-001…RAG-009 | `done` em `.roadmap/` (ESAA) |
| Providers | `hash`, `ollama` |
| Schema DB | v2 (fingerprint, gerações, BLOB f32) |

## License / Provenance

Código e documentação do workspace do operador. Referência de desenho de
retrieval alinhada ao `flask-dashboard` (`rag.py` / `TextEmbeddingService`).
Governança de tarefas alinhada ao protocolo ESAA.
