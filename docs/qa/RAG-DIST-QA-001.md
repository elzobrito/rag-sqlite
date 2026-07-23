# RAG-DIST-QA-001 — Release bundle validation

Release candidate: `rag-sqlite-v0.1.0.zip`

## Result

Approved for publication after the automated checks below pass from a clean
temporary directory.

## Coverage

- deterministic archive bytes and external checksum;
- exact allowlist of runtime payload files;
- executable Linux launcher and static Windows launcher contract;
- rejection policy for absolute paths, traversal entries, and symbolic links;
- internal checksum verification and corruption rejection;
- clean extraction followed by `schema`;
- offline `hash` provider indexing and query without Ollama;
- absence of databases, caches, environment files, event stores, and user data.

## Platform note

The Linux launcher is executed by the local smoke test. The Windows launcher is
validated structurally here; native Windows execution belongs to the
Conversation ESAA installer matrix, where the bundle is consumed.
