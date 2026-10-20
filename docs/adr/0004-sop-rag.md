# ADR 0004: RAG over SOPs for alert guidance

**Status:** accepted

**Context.** Alerts say *what* happened ("unattended bag in Check-in Hall") but operators need *what to
do*. Procedures live in documents and differ by airport, so they must not be hard-coded.

**Decision.** Retrieve from a folder of markdown SOPs. Each document declares `alert_types` in
front-matter; documents are chunked by `##` section. Retrieval = BM25 + a domain synonym map + a
metadata boost for the alert's type. Optional `hybrid` backend adds local sentence-transformers
embeddings. Retrieved steps are attached to the alert with cited sources. Free-text questions
(`/api/sop/ask`) are answered only from retrieved passages; an LLM (Gemini free tier) may phrase the
answer if a key is set, otherwise the best document's steps are returned verbatim.

**Why not LangChain / a vector DB?** The corpus is tens of chunks. A dependency-free retriever is
easier to test, runs offline, and has nothing to configure. The `SopIndex` interface (`search`,
`for_alert`, `ask`) is small enough to swap for a vector store when the corpus grows.

**Safety.** Output is advisory only, always cites sources, and is never wired to automatic actions.
If nothing relevant is retrieved the answer says so instead of improvising.

**Consequences.** Lexical retrieval is brittle to vocabulary mismatch (e.g. "fell" vs "fall"); a
synonym map patches common cases and `hybrid` embeddings are the proper fix. The bundled SOPs are
*samples*, not real procedures; replace them before any operational use.
