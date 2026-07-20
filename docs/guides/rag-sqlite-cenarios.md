# rag-sqlite — Cenários práticos (cookbook)

Este guia documenta os comandos do `rag_sqlite.py` **dentro de situações reais**,
não como lista seca de flags. Cada cenário tem objetivo, comandos na ordem de
uso, saída esperada e armadilhas.

Estilo alinhado aos [Cenários práticos do ESAA-Core](../../../ESAA-Core/docs/guides/esaa-cenarios.md).

Para sintaxe exaustiva: [Referência do CLI](rag-sqlite-cli-reference.md).  
Para o caminho mais curto: [Primeiros passos](rag-sqlite-getting-started.md).

> Convenções: exemplos em **bash**. Substitua `./kb.sqlite` pelo path do DB.
> Toda saída útil está em **stdout como JSON**; logs opcionais em stderr com
> `--verbose`.

## Índice

1. [Base offline em 60 segundos](#cenário-1--base-offline-em-60-segundos)
2. [Filtrar só hits bons](#cenário-2--filtrar-só-hits-bons)
3. [Exportar contexto para o LLM](#cenário-3--exportar-contexto-para-o-llm)
4. [Ativar Ollama local e reindexar](#cenário-4--ativar-ollama-local-e-reindexar)
5. [Ollama em servidor remoto](#cenário-5--ollama-em-servidor-remoto)
6. [Trocar modelo sem corromper o rank](#cenário-6--trocar-modelo-sem-corromper-o-rank)
7. [Indexar pasta e sincronizar deletes](#cenário-7--indexar-pasta-e-sincronizar-deletes)
8. [Travar paths e hosts (segurança)](#cenário-8--travar-paths-e-hosts-segurança)
9. [Um arquivo ruim não derruba o lote](#cenário-9--um-arquivo-ruim-não-derruba-o-lote)
10. [Agente descobre a API sozinho](#cenário-10--agente-descobre-a-api-sozinho)
11. [Health ready / degraded / unhealthy](#cenário-11--health-ready--degraded--unhealthy)
12. [Governança ESAA no mesmo workspace](#cenário-12--governança-esaa-no-mesmo-workspace)

---

## Cenário 1 — Base offline em 60 segundos

**Situação:** validar o pipeline sem rede e sem Ollama.

```bash
cd /home/elzobrito/desenvolvimento/rag-sqlite

python rag_sqlite.py --db ./kb.sqlite config set embedding_provider hash
python rag_sqlite.py --db ./kb.sqlite index ./tests/fixtures
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --top-k 3
```

**Esperado:**

- `config set`: `db_created: true` na primeira execução
- `index`: `totals.indexed == 2` (alpha + beta)
- `query`: primeiro hit `alpha.txt`, `hybrid_alpha: 0.0`, `provider: hash`

**Armadilha:** provider `hash` não é semântico; serve para testes e lexical.

---

## Cenário 2 — Filtrar só hits bons

**Situação:** beta (jardinagem) aparece com score 0 e polui o context.

```bash
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --min-score 0.1
```

**Esperado:** `hit_count: 1`, só `alpha.txt`.

Alternativa relativa (mantém hits próximos do top-1):

```bash
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --min-score-relative 0.85
```

---

## Cenário 3 — Exportar contexto para o LLM

**Situação:** o agente só precisa de texto para colar no prompt, não do JSON cheio.

```bash
python rag_sqlite.py --db ./kb.sqlite --compact export-context "data mesh" \
  --top-k 5 --min-score 0.1
```

**Esperado:** schema `rag_sqlite.export_context.v1` com campos `context`,
`hit_count`, `meta`.

**Armadilha:** sem `--min-score`, o export usa o default do settings (`0.0`) e
pode incluir hits fracos — passe os mesmos filtros da query.

---

## Cenário 4 — Ativar Ollama local e reindexar

**Situação:** passar de teste lexical para embeddings reais.

```bash
# daemon Ollama + modelo já baixado
# ollama pull embeddinggemma

python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url http://127.0.0.1:11434 \
  --model embeddinggemma

python rag_sqlite.py --db ./kb.sqlite health
python rag_sqlite.py --db ./kb.sqlite reindex --force
python rag_sqlite.py --db ./kb.sqlite query "data mesh" --min-score 0.4
```

**Esperado:**

- `index_fingerprint` **diferente** do modo hash
- `dimensions: 768` (embeddinggemma típico)
- `generation_id` novo, `generation_activated: true`
- `hybrid_alpha: 0.7`, alpha na frente de beta por cosine+keyword

**Armadilha:** sem `reindex --force` após `set-ollama`, a query pode achar
`candidate_count: 0` (fingerprint atual sem chunks ativos).

---

## Cenário 5 — Ollama em servidor remoto

**Situação:** o embed roda em outra máquina ou URL HTTPS.

```bash
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url https://ollama.exemplo.com \
  --model embeddinggemma \
  --timeout 180

python rag_sqlite.py --db ./kb.sqlite config get base_url
python rag_sqlite.py --db ./kb.sqlite health
python rag_sqlite.py --db ./kb.sqlite reindex --force
```

**Esperado:** `settings.base_url` normalizado (sem `/` final); health com
`provider_ok` se a URL responder em `/api/tags`.

**Armadilha:** hosts HTTPS com certificado problemático dependem da política
SSL do Python stdlib — falha vira `OllamaConnectionError` / HTTP error em JSON.

---

## Cenário 6 — Trocar modelo sem corromper o rank

**Situação:** mudar de `embeddinggemma` para outro modelo no mesmo host.

```bash
python rag_sqlite.py --db ./kb.sqlite config set embedding_model nomic-embed-text
python rag_sqlite.py --db ./kb.sqlite stats   # fingerprint novo
python rag_sqlite.py --db ./kb.sqlite reindex --force
python rag_sqlite.py --db ./kb.sqlite query "data mesh"
```

**Esperado:** `stats.index_fingerprint` muda; query usa só a geração ativa do
fingerprint corrente.

**Armadilha:** duas gerações `active` de **fingerprints diferentes** podem
aparecer em `meta.generations` — isso é por-fingerprint; a query não mistura.

---

## Cenário 7 — Indexar pasta e sincronizar deletes

**Situação:** removeu um `.md` do disco e quer alinhar o inventário.

```bash
python rag_sqlite.py --db ./kb.sqlite index ./corpus --sync
# inspecione sync.missing

python rag_sqlite.py --db ./kb.sqlite index ./corpus --sync --prune
```

**Esperado:**

- `--sync` lista `missing` sem apagar
- `--prune` remove do SQLite apenas com flag explícita

**Armadilha:** nunca há delete automático só por indexar de novo.

---

## Cenário 8 — Travar paths e hosts (segurança)

**Situação:** um agente não deve indexar `/etc` nem chamar Ollama arbitrário.

```bash
python rag_sqlite.py --db ./kb.sqlite config set index_root /home/elzobrito/desenvolvimento/rag-sqlite
python rag_sqlite.py --db ./kb.sqlite config set allowed_hosts "127.0.0.1,localhost"
python rag_sqlite.py --db ./kb.sqlite config set allow_symlinks false

# deve falhar
python rag_sqlite.py --db ./kb.sqlite index /tmp/segredo.txt
python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url https://evil.example.com --model x
```

**Esperado:** `PathOutsideRoot`, `HostNotAllowed` em `rag_sqlite.error.v1`.

---

## Cenário 9 — Um arquivo ruim não derruba o lote

**Situação:** pasta com um binário e vários textos bons.

```bash
python rag_sqlite.py --db ./kb.sqlite config set max_file_bytes 100000
python rag_sqlite.py --db ./kb.sqlite index ./pasta_mista
```

**Esperado:** entradas `status: error` por arquivo; demais `indexed`;
SAVEPOINT isola a falha.

Arquivos acima de `max_file_bytes` entram como erro de arquivo, não crash do CLI.

---

## Cenário 10 — Agente descobre a API sozinho

**Situação:** tool-calling sem documentação embutida no prompt longo.

```bash
python rag_sqlite.py schema
python rag_sqlite.py schema query
python rag_sqlite.py schema "config set-ollama"
```

**Esperado:** `json_schema` com `args.properties`, `required`, enums de health
quando aplicável.

Padrão mental igual a ler `dispatch-context` no ESAA antes de emitir envelope:
**descubra o contrato, depois invoque.**

---

## Cenário 11 — Health ready / degraded / unhealthy

```bash
python rag_sqlite.py --db ./kb.sqlite config set embedding_provider hash
python rag_sqlite.py --db ./kb.sqlite health
# status: ready

python rag_sqlite.py --db ./kb.sqlite config set-ollama \
  --url http://127.0.0.1:11434 --model modelo-que-nao-existe
python rag_sqlite.py --db ./kb.sqlite health
# status: degraded se tags ok mas model ausente; unhealthy se conexão falha
```

Probe opcional de embed:

```bash
python rag_sqlite.py --db ./kb.sqlite config set health_probe_embed true
python rag_sqlite.py --db ./kb.sqlite health
```

---

## Cenário 12 — Governança ESAA no mesmo workspace

**Situação:** mudanças de produto sob claim/complete/review (política do
operador + [guias ESAA](../../../ESAA-Core/docs/guides/esaa-cenarios.md)).

```bash
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite verify
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite eligible

# ciclo resumido (ver ESAA getting-started para detalhes)
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite --runner grok \
  claim RAG-XXX --actor agent-impl
# ... implementar ...
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite --runner grok \
  complete RAG-XXX --actor agent-impl --check "..."
esaa --root /home/elzobrito/desenvolvimento/rag-sqlite --runner grok \
  review RAG-XXX --actor agent-qa --decision approve --review-mode functional
```

**Armadilha:** `rag_sqlite.py` **não** substitui ESAA. Conversation ESAA
(handoff) também não. Cada sistema tem domínio próprio.

---

## Referência rápida comando → cenário

| Comando | Cenários |
|---------|----------|
| `config set` / `set-ollama` | 1, 4, 5, 6, 8 |
| `index` / `reindex` | 1, 4, 6, 7, 9 |
| `query` / `export-context` | 1, 2, 3, 4 |
| `health` / `stats` | 4, 11 |
| `schema` | 10 |
| `esaa *` | 12 |

## Mapa de troubleshooting

| Erro (`error.type`) | Cenário / ação |
|---------------------|----------------|
| `DB_NOT_FOUND` | Criar via write ou `--create` |
| `HostNotAllowed` | #8 |
| `PathOutsideRoot` | #8 |
| `DimensionMismatch` | `reindex --force` |
| `OllamaConnectionError` | #4 / #5 + `health` |
| `UsageError` | flags globais antes do subcomando |
| `unknown_setting` | `config list` |
