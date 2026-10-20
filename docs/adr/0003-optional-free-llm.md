# ADR 0003: LLM is optional, free-tier and text-only

**Status:** accepted

**Context.** Natural-language search is valuable, but paid APIs and sending video to third parties are
non-starters here.

**Decision.** Default to a deterministic rule parser. If `GEMINI_API_KEY` (free tier) is set, use the
model only to translate the question into a structured query; on any failure fall back to rules. Only
the question and zone names leave the process.

**Consequences.** Works offline with no keys. The LLM can't hallucinate numbers because answers always
come from the database, not from the model.
