# Piper AI Agent — System Architecture

> A production-grade, multi-agent customer support system built on gRPC microservices, ReACT reasoning, and persistent learning through Reflexion.

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [Service Topology](#2-service-topology)
3. [The 15-Stage Query Pipeline](#3-the-15-stage-query-pipeline)
4. [Detailed Stage Walkthrough](#4-detailed-stage-walkthrough)
5. [ReACT Reasoning Engine](#5-react-reasoning-engine)
6. [Semantic Search — How Products Are Found](#6-semantic-search--how-products-are-found)
7. [Multi-Agent Orchestration](#7-multi-agent-orchestration)
8. [Reflection vs Reflexion — The Two Learning Systems](#8-reflection-vs-reflexion--the-two-learning-systems)
9. [Recommendation Engine](#9-recommendation-engine)
10. [Data Storage Architecture](#10-data-storage-architecture)
11. [End-to-End Query Walkthrough](#11-end-to-end-query-walkthrough)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Security & Authentication](#13-security--authentication)
14. [Resilience Patterns](#14-resilience-patterns)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. High-Level System Overview

The Piper AI Agent is a microservices-based conversational AI system where a central **Agent Service** orchestrates reasoning, tool execution, quality assurance, and persistent learning across specialized services connected via gRPC.

### Technology Stack

| Component | Technology |
|-----------|------------|
| Session cache | Redis 7 (TTL: 30 min sliding) |
| Relational storage | PostgreSQL 16 + pgvector |
| Immutable store | TimescaleDB (episodic memory + audit trail) |
| Vector search | pgvector (1024-dim Voyage AI embeddings) |
| LLM | Anthropic Claude Sonnet |
| Embeddings | Voyage AI voyage-3 |
| Services | Python gRPC + FastAPI |
| Authentication | JWT (HS256) + bcrypt passwords |
| Transport Security | TLS (self-signed CA for dev) |
| Containers | Docker Compose |
| Resilience | tenacity (retry / circuit breaker) |
| Logging | structlog (structured JSON) |

![High-Level System Overview](images/01-high-level-system-overview.svg)

---

## 2. Service Topology

Each service runs as an independent gRPC server. All inter-service communication uses Protocol Buffers.

![Proto Contracts](images/02-proto-contracts.svg)

| Service                    | Address             | Role                                                      |
| -------------------------- | ------------------- | --------------------------------------------------------- |
| **Gateway Server**         | `:8765` (WebSocket) | Client-facing entry point, JWT auth, rate limiting        |
| **Agent Service**          | `:50054` (gRPC)     | Core orchestrator — 15-stage pipeline, ReACT, multi-agent |
| **Memory Service**         | `:50055` (gRPC)     | Session state, conversation history, episodic memories    |
| **Tool Service**           | `:50056` (gRPC)     | Domain tool execution with schema validation              |
| **LLM Service**            | `:50053` (gRPC)     | Claude API wrapper with temperature/token routing         |
| **Knowledge Service**      | `:50052` (gRPC)     | Voyage AI embeddings + semantic product search            |
| **Recommendation Service** | `:50057` (gRPC)     | Context-aware suggestion generation                       |

### Request Flow Summary

![Request Flow Summary](images/03-request-flow-summary.svg)

---

## 3. The 15-Stage Query Pipeline

Every user query flows through a deterministic 15-stage pipeline inside `ProcessQuery()`. Each stage is independently feature-flagged.

![15-Stage Query Pipeline](images/04-twelve-stage-pipeline.svg)

---

## 4. Detailed Stage Walkthrough

### Stage 0 — Session Context & Structured Memory

```
Session Touch (Redis TTL refresh)
    +-- Load last 10 conversation turns from PostgreSQL
    +-- Build structured memory_context with 3 sections:
    |     1. Conversation Flow (numbered exchange summaries)
    |     2. Active Context (products mentioned, intents seen, tools called)
    |     3. Latest Exchange (full detail for current topic)
    +-- Extract previous turn's intent + tool_calls for follow-up detection
    +-- Store current user query as new conversation turn
```

The structured memory context replaces the flat "User: ... / Assistant: ..." dump used previously. Older assistant responses are truncated to `MEMORY_CONTEXT_TRUNCATE_LENGTH` (default 200 chars); the latest exchange is preserved in full. Products are detected by matching against a known catalog product list (`_CATALOG_PRODUCTS`).

### Stage 1 — Input Guardrails

Three regex-based checks run on every input. No LLM calls are made — this is pure pattern matching for speed.

| Check                   | Pattern                       | Action                                     |
| ----------------------- | ----------------------------- | ------------------------------------------ |
| **Query length**        | `len(query) > 2000`           | Block with `guardrail_blocked`             |
| **Injection detection** | 5 compiled regex patterns     | Block with `guardrail_blocked`             |
| **PII warning**         | 4 compiled regex patterns     | Log warning, continue with sanitized input |

**Injection patterns detected:**
- `ignore (all) previous instructions`
- `you are now a/an ...`
- `system:` prefix
- `<system>`, `<admin>`, `<root>` tags
- `forget/disregard everything/all/your`

**PII patterns detected (warning only):** Email addresses, phone numbers, SSN format, credit card numbers.

**Implementation**: `_check_input_guardrails(query)` returns `(is_safe, sanitized_query, issues)`. When blocked, `ProcessQuery()` yields a `guardrail_blocked` event and returns immediately — no further processing occurs.

### Stage 2 — Query Rewriting (Pronoun Resolution)

An LLM call rewrites the user's query to be fully self-contained by resolving pronouns, demonstratives, and ellipsis using conversation history.

| Input                                    | Rewritten Output                               |
| ---------------------------------------- | ---------------------------------------------- |
| "How much does it cost?"                 | "How much does the RoboCleaner 3120 cost?"     |
| "What about the cheaper one?"            | "What about the EcoKettle 1042?"               |
| "Compare them"                           | "Compare the UltraWasher 8262 and PowerDrill 5641" |
| "Tell me more"                           | "Tell me more about the RoboCleaner 3120"      |

**Guard rails on the rewrite:** Rewrites are rejected (falling back to the original query) if the result is empty, if it exceeds 5x the original query length (with a floor of 200 chars), or if the LLM call fails for any reason.

**Feature flag:** `QUERY_REWRITE_ENABLED` (default `true`). Only runs when conversation history exists (first query in a session is never rewritten).

**Implementation:** `_rewrite_query(query, history_turns)` uses `QUERY_REWRITE_PROMPT` with `temp=0.1, max_tokens=128`.

### Stage 3 — Intent Classification

The LLM classifies the user's query into one of 9 intent categories, now with **domain relevance scoring** and **previous turn awareness**:

| Intent              | Description                          | Routing                     |
| ------------------- | ------------------------------------ | --------------------------- |
| `product_inquiry`   | Questions about product details      | ReACT or Multi-Agent        |
| `price_check`       | Price lookups and budget queries     | ReACT or Multi-Agent        |
| `comparison`        | Comparing products/brands            | ReACT or Multi-Agent        |
| `warranty_question` | Warranty coverage and claims         | ReACT or Multi-Agent        |
| `follow_up`         | Continuing previous conversation     | ReACT or Multi-Agent        |
| `session_query`     | Meta-queries about the conversation  | Direct history lookup       |
| `general_question`  | General chat, greetings              | Direct LLM (no tools)       |
| `out_of_scope`      | Off-topic queries                    | Catalog-aware redirect      |

**New fields in classification output:**

- `domain_relevance` (0.0-1.0): How related the query is to the product catalog. HIGH (0.7-1.0) for product topics, LOW (0.0-0.3) for unrelated topics.
- `previous_intent` and `previous_entities`: Injected into the prompt so the LLM can detect follow-ups and continuations.

**Follow-up detection:** When the previous turn was product-related and the current query references or continues that discussion (e.g., "Tell me more", "What about the warranty?"), the classifier assigns `follow_up` with high confidence and high domain relevance.

### Stage 4 — Decision Routing

After classification, the pipeline routes the query through one of four paths:

**Path 0 — Session Query:** If `intent == "session_query"`, the system answers directly from conversation history without any LLM or tool calls. Extracts all user queries from `history_turns`, formats them as a numbered list, applies output guardrails, and streams the response.

**Path A — Domain Relevance Redirect:** If `domain_relevance < DOMAIN_RELEVANCE_THRESHOLD` (default 0.5), the system responds with a catalog-aware redirect message listing all product categories and capabilities. No ReACT loop is executed.

**Path B — Dynamic Clarification Gate:** If `needs_clarification = true`, `confidence < INTENT_CONFIDENCE_THRESHOLD`, and `intent != "follow_up"`, the system sends a structured clarification request. Clarification options are now **context-aware** based on entity keywords:

| Detected Category | Keywords                                        | Options Offered                          |
| ----------------- | ----------------------------------------------- | ---------------------------------------- |
| Cleaning          | clean, vacuum, wash, mop, floor, dust           | RoboCleaner, SuperVac, UltraWasher       |
| Kitchen           | blend, cook, kitchen, kettle, boil, smoothie    | MegaBlender, EcoKettle                   |
| Smart Home        | smart, lamp, light, air, noise, purif           | SmartLamp, AirPurifier, NoiseCanceller   |
| *(fallback)*      | No category match                               | Generic intent-based options             |

The clarification turn is stored to memory before yielding (ensuring conversation history stays complete). The user's response can include enriched product context (e.g., selecting "RoboCleaner" appends `-- I'm looking for product recommendations about RoboCleaner` to the original query).

**Path C — Normal Processing:** High domain relevance + high confidence. Proceeds to Planning (Stage 5).

### Stage 5 — Planning Layer

The LLM decomposes the query into executable sub-goals:

```json
{
  "needs_multi_agent": true,
  "plan_steps": [
    {
      "goal": "Look up UltraWasher warranty",
      "suggested_tool": "warranty_check",
      "priority": 1
    },
    {
      "goal": "Look up RoboCleaner warranty",
      "suggested_tool": "warranty_check",
      "priority": 1
    },
    { "goal": "Compare warranty terms", "suggested_tool": null, "priority": 2 }
  ],
  "specialist_agents": ["warranty_specialist", "comparison_specialist"]
}
```

This plan is injected into the ReACT system prompt as an execution guide, and determines whether single-agent or multi-agent orchestration is used.

### Stage 10 — Output Guardrails (PII Redaction)

The same PII regex patterns from Stage 1 are applied to the response text. Unlike input guardrails, output guardrails **redact** rather than block:

| PII Type    | Example              | Redaction          |
|-------------|----------------------|--------------------|
| Email       | `user@example.com`   | `[EMAIL REDACTED]` |
| Phone       | `(555) 123-4567`     | `[PHONE REDACTED]` |
| SSN         | `123-45-6789`        | `[SSN REDACTED]`   |
| Credit Card | `1234 5678 9012 3456`| `[CARD REDACTED]`  |

`_check_output_guardrails(response_text)` returns `(sanitized_text, was_modified, redactions)`. If PII is redacted, a `guardrail_sanitized` event is emitted.

### Stage 12 — Evaluation Storage

Every request stores a structured evaluation record to TimescaleDB as an episodic memory with `event_type = 'evaluation_record'`:

```json
{
    "query": "Compare warranty of Product A vs Product B",
    "intent": "comparison",
    "confidence": 0.85,
    "reflection_score": 0.78,
    "tools_used": ["warranty_check", "warranty_check"],
    "reasoning_steps": 4,
    "latency_ms": 3200,
    "response_length": 450
}
```

**Analytics query examples:**
```sql
-- Average latency by intent
SELECT metadata->>'intent' AS intent,
       AVG((metadata->>'latency_ms')::int) AS avg_latency_ms,
       AVG((metadata->>'confidence')::float) AS avg_confidence
FROM episodic_memories
WHERE event_type = 'evaluation_record'
GROUP BY metadata->>'intent';

-- Low-confidence responses in last 24h
SELECT * FROM episodic_memories
WHERE event_type = 'evaluation_record'
  AND (metadata->>'confidence')::float < 0.6
  AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
```

`_store_evaluation_record()` is non-blocking — failures are logged and swallowed.

### Pipeline Latency Budget

| Component | Extra LLM Calls | Approximate Overhead |
|---|---|---|
| Input guardrails (regex) | 0 | <1ms |
| Query rewriting (pronoun resolution) | +1 | ~0.5-1s |
| Intent classification (with domain relevance) | +1 | ~1-2s |
| Planning | +1 | ~1-2s |
| Multi-agent (2 specialists + synthesis) | +3-5 | ~5-10s |
| Reflection: passes (score >= 0.75) | +1 | ~1-2s |
| Reflection: 1 refinement | +2 | ~2-4s |
| Reflection: 2 refinements (max) | +4 | ~4-8s |
| Reflexion insight stored | +1 | ~1-2s |
| Output guardrails (regex) | 0 | <1ms |
| Evaluation storage (DB write) | 0 | ~50ms |

Worst case (query rewrite + planning + multi-agent + 2 reflection refinements + reflexion): ~16-22s additional. Well within the 120s timeout.

### Pipeline Methods Reference

All methods on `AgentServiceServicer` in `agent_service/server.py`:

| Method | Stage | Purpose |
|---|---|---|
| `ProcessQuery()` | -- | Main entry point; orchestrates all stages |
| `build_memory_context(turns)` | 0 | Build structured 3-section context from conversation turns |
| `_check_input_guardrails(query)` | 1 | Regex PII + injection detection on input |
| `_rewrite_query(query, history_turns)` | 2 | LLM call to resolve pronouns/references in the query |
| `_classify_intent(query, context, previous_intent, previous_entities)` | 3 | LLM call for intent + domain relevance classification |
| `_handle_session_query(session_id, ...)` | 4.0 | Answer meta-queries from conversation history |
| `_handle_out_of_scope_redirect(session_id, ...)` | 4.A | Catalog-aware redirect for low domain relevance |
| `_build_clarification_options(intent_result)` | 4.B | Build context-aware clarification options |
| `_generate_plan(query, intent, tool_list, context)` | 5 | LLM call to decompose query into sub-goals |
| `_execute_react_step(query, memory, tools, history, intent)` | 6 | Single ReACT iteration via LLM |
| `_validate_tool_params(tool_name, params, schema)` | 6 | Validate tool call parameters against schema |
| `_execute_tool(session_id, tool_name, params)` | 6 | Execute tool via Tool Service gRPC |
| `_validate_tool_result(tool_name, result)` | 6 | Validate tool output completeness |
| `_get_reflexion_insights(customer_id, intent, query)` | 6 | Fetch past learnings from episodic memory |
| `_run_react_loop(session_id, ...)` | 6 | Full ReACT loop orchestration |
| `_run_agent_sub_loop(agent_type, query, ...)` | 7 | Focused 4-iteration ReACT loop for specialist |
| `_run_multi_agent_loop(session_id, ...)` | 7 | Orchestrate sequential specialist agents |
| `_synthesize_multi_agent_response(results, query)` | 7 | LLM call to combine specialist outputs |
| `_frame_response(query, answer, tools, steps, memory_context)` | 8 | LLM call to polish answer with conversation context |
| `_evaluate_response(query, text, tools, steps, ctx)` | 9 | LLM call to score response quality |
| `_refine_response(query, text, tools, eval, obs, memory_context)` | 9 | LLM call to improve response with conversation context |
| `_run_reflection_loop(query, framed, tools, steps, ctx)` | 9 | Evaluate-refine loop orchestration |
| `_check_output_guardrails(response_text)` | 10 | PII redaction on output |
| `_generate_reflexion_insight(query, intent, ...)` | 11 | LLM call to produce reusable learning |
| `_maybe_store_reflexion_insight(session_id, ...)` | 11 | Store insight if quality below threshold |
| `_store_evaluation_record(session_id, ...)` | 12 | Store evaluation metrics to TimescaleDB |
| `_handle_simple_intent(session_id, ...)` | 3->10->13 | Handle general_question directly |

---

## 5. ReACT Reasoning Engine

The core reasoning loop implements the **ReACT** (Reasoning + Acting) framework. The agent alternates between thinking and tool execution until it has enough information to answer.

![ReACT Reasoning Engine](images/05-react-reasoning-engine.svg)

### ReACT Output Format

Each LLM call produces exactly one of two patterns:

**Pattern 1 — Need more information:**

```
Thought: I need to check the warranty for UltraWasher 8262.
Action: warranty_check({"product_name": "UltraWasher 8262"})
```

**Pattern 2 — Ready to answer:**

```
Thought: I now have warranty details for both products. UltraWasher has
         24 months and RoboCleaner has 36 months.
Answer: The RoboCleaner 3000 has a longer warranty at 36 months compared
        to UltraWasher 8262's 24-month warranty.
```

### Tool Validation Pipeline

Every tool call passes through a two-phase validation gate:

![Tool Validation Pipeline](images/06-tool-validation-pipeline.svg)

Invalid parameters are **not blocked** — they're returned as observations so the LLM can self-correct on the next iteration.

### System Prompt Assembly

Before the loop begins, the system constructs a composite prompt from four injected sections. This prompt stays constant across all iterations — only the user prompt (with accumulating history) changes per iteration.

![System Prompt Assembly](images/07-system-prompt-assembly.svg)

The assembled system prompt sent to the LLM:

```
You are Piper, an AI customer support agent for a product catalog...
You must reason step-by-step using the ReACT framework.

Learnings from past interactions (use these to improve your response):
- When comparing warranties, always check both products before answering

Execution plan (follow these steps):
  1. Look up UltraWasher warranty (use warranty_check)
  2. Look up price (use price_lookup)

Available tools:
- product_search(query, top_k): Search products by description
- price_lookup(product_name): Look up price for a product
- warranty_check(product_name): Check warranty details
- product_compare(product_names): Compare products side-by-side

Rules:
- Always think before acting
- Use tools to get factual information; do not make up product details
- Format your response as exactly one of these two patterns:
  Pattern 1: Thought: [...] Action: tool_name({...})
  Pattern 2: Thought: [...] Answer: [...]
```

### Step-by-Step Iteration Trace

The following traces a real query through two iterations: **"What's the warranty on UltraWasher 8262 and how much does it cost?"**

![Step-by-Step Iteration Trace](images/08-iteration-trace.svg)

### How History Accumulates

The critical mechanism is `build_react_history()` — each iteration's thought, action, and observation are appended to a growing history string. On every subsequent LLM call, the model sees **all** previous reasoning, giving it memory within the loop.

![How History Accumulates](images/09-history-accumulation.svg)

Each step entry has a fixed structure:

| Field          | Content                                                 |
| -------------- | ------------------------------------------------------- |
| `iteration`    | Iteration number (1, 2, 3...)                           |
| `thought`      | What the LLM reasoned                                   |
| `action`       | Tool name it chose (or absent if it gave an Answer)     |
| `action_input` | JSON parameters sent to the tool                        |
| `observation`  | Raw tool result, or error message, or enriched guidance |

### Self-Correction Mechanisms

The ReACT loop has five built-in recovery paths. None of them crash the loop — they all produce an observation that guides the LLM to fix its own mistake.

![Self-Correction Mechanisms](images/10-self-correction-mechanisms.svg)

**Example: Self-correction in action over two iterations**

```
Iteration 3:
  Thought: I need to check the warranty for RoboCleaner.
  Action: warranty_check({"product": "RoboCleaner 3000"})
                          ^^^^^^^^^ wrong field name

  Parameter validation fails:
    "Missing required field 'product_name' for tool warranty_check"

  Observation stored: "Parameter validation error: Missing required
                       field 'product_name'. Please fix and try again."

Iteration 4 (LLM sees the error in history):
  Thought: I used the wrong parameter name. The field should be
           'product_name', not 'product'.
  Action: warranty_check({"product_name": "RoboCleaner 3000"})
                          ^^^^^^^^^^^^^ corrected

  Validation passes. Tool executes successfully.
```

### After the Loop — Response Framing

Once `final_answer` is set, the raw answer is passed to `_frame_response()` — a separate LLM call that polishes the agent's internal reasoning into a user-facing response:

| Input             | Value                                                                |
| ----------------- | -------------------------------------------------------------------- |
| `query`           | "What's the warranty on UltraWasher 8262 and how much does it cost?" |
| `answer`          | Raw answer from the ReACT loop                                       |
| `tools_used`      | `["warranty_check"]`                                                 |
| `reasoning_steps` | `2` (number of iterations)                                           |

The framing LLM returns structured JSON:

```json
{
  "text": "The UltraWasher 8262 is priced at $121.24 and comes with a 6-month
           manufacturer's warranty starting from its manufacturing date of
           September 21, 2023. This means the warranty coverage extends
           through March 2024.",
  "confidence": 0.93,
  "sources": ["UltraWasher 8262"]
}
```

This framed response then continues through Reflection (Stage 9), Output Guardrails (Stage 10), Reflexion (Stage 11), and finally streams to the client.

### Design Principles

The ReACT loop enforces a strict separation between reasoning and data access:

| Concern                      | Who Handles It                  |
| ---------------------------- | ------------------------------- |
| **Deciding** what to look up | LLM (Thought)                   |
| **Requesting** a tool call   | LLM (Action)                    |
| **Validating** the request   | Agent Service (param validator) |
| **Executing** the request    | Tool Service (gRPC)             |
| **Providing** ground truth   | PostgreSQL / Knowledge Service  |
| **Interpreting** the result  | LLM (next Thought)              |
| **Deciding** when to stop    | LLM (Answer)                    |

The LLM never has direct database access. It can only invoke pre-defined tools with validated parameters and observe structured results. This controlled loop prevents hallucination of product data — the LLM must cite what the tools returned, not what it imagines.

---

## 6. Semantic Search — How Products Are Found

When the ReACT engine calls `product_search`, the query doesn't do keyword matching — it uses **vector embeddings** to find products by semantic meaning. A query like _"affordable washing machine with good durability"_ finds products even if those exact words never appear in the product name.

### Embedding Model and Storage

| Component             | Technology                                     |
| --------------------- | ---------------------------------------------- |
| **Embedding model**   | Voyage AI `voyage-3`                           |
| **Vector dimensions** | 1024 floats per embedding                      |
| **Vector storage**    | PostgreSQL with `pgvector` extension           |
| **Index type**        | IVFFlat with `vector_cosine_ops` (10 clusters) |
| **Distance metric**   | Cosine distance via `<=>` operator             |

### Ingestion Pipeline — Pre-Computing Product Embeddings

At system startup, `scripts/seed_products.py` embeds every product in the catalog. Each product is converted to a rich text string before embedding, combining name, description, price, and warranty into a single semantic representation:

![Ingestion Pipeline](images/11-ingestion-pipeline.svg)

Products are processed in batches of 10 with a 21-second pause between batches (Voyage AI free-tier limit: 3 requests per minute). The script tracks which products already have embeddings to support resume-on-failure.

### Query-Time Search — The Full Chain

When a user asks _"Tell me about UltraWasher"_, the ReACT engine calls `product_search`. Here is the exact path through all four services:

![Query-Time Search Chain](images/12-query-time-search.svg)

### The Cosine Similarity Math

The pgvector `<=>` operator computes cosine distance between the query vector and each stored product vector:

```
cosine_distance(A, B) = 1 - (A . B) / (||A|| x ||B||)
```

The SQL then converts distance to a similarity score:

```sql
1 - (pe.embedding <=> query_vector) AS similarity
```

![Cosine Similarity Math](images/13-cosine-similarity.svg)

| Score Range     | Meaning               | Example                                            |
| --------------- | --------------------- | -------------------------------------------------- |
| **0.90 - 1.00** | Near-exact match      | "UltraWasher" vs "UltraWasher 8262"                |
| **0.60 - 0.89** | Strong semantic match | "affordable washing machine" vs "UltraWasher 8262" |
| **0.30 - 0.59** | Weak relevance        | "kitchen appliance" vs "UltraWasher 8262"          |
| **0.00 - 0.29** | Unrelated             | "power drill" vs "UltraWasher 8262"                |

### Why Semantic Search Matters

Unlike SQL `LIKE '%UltraWasher%'` (keyword matching), semantic search captures **meaning**:

| User Query                              | Keyword Match    | Semantic Match                                                   |
| --------------------------------------- | ---------------- | ---------------------------------------------------------------- |
| "UltraWasher"                           | UltraWasher 8262 | UltraWasher 8262                                                 |
| "affordable washing machine"            | No results       | UltraWasher 8262 (via "washing" + low price in description)      |
| "durable cleaning appliance"            | No results       | UltraWasher 8262 (via "durability" in description)               |
| "something to clean clothes under $150" | No results       | UltraWasher 8262 (semantic proximity to washing + price context) |

This is possible because the pre-computed embedding captures the full semantic meaning of `"UltraWasher 8262: Offers superior performance and durability. Price: $121.24. Warranty: 6 months."` — not just the words, but their contextual relationships.

### Database Schema

```sql
-- pgvector extension
CREATE EXTENSION IF NOT EXISTS "vector";

-- Embedding storage
CREATE TABLE product_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index for fast cosine similarity search
CREATE INDEX idx_product_embeddings_ivfflat
    ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
```

The IVFFlat index partitions vectors into 10 clusters, allowing PostgreSQL to search only the most relevant clusters instead of scanning all vectors — reducing query time from O(n) to approximately O(n/10) for the 1024-dimensional space.

---

## 7. Multi-Agent Orchestration

When the Planning Layer determines a query requires multiple domains, specialist agents are dispatched in sequence. Each runs a focused ReACT sub-loop, and outputs are synthesized.

![Multi-Agent Orchestration](images/14-multi-agent-orchestration.svg)

### Agent Registry

| Agent                     | Focus                                     | Preferred Tools                                     |
| ------------------------- | ----------------------------------------- | --------------------------------------------------- |
| **product_specialist**    | Product details, features, specifications | `product_search`, `price_lookup`                    |
| **warranty_specialist**   | Warranty policies, claims, coverage       | `warranty_check`                                    |
| **comparison_specialist** | Systematic multi-dimension comparison     | `product_search`, `price_lookup`, `product_compare` |

Each specialist receives a modified system prompt with their focus area and preferred tools listed first, biasing the LLM toward domain-relevant tool usage.

---

## 8. Reflection vs Reflexion — The Two Learning Systems

These are two architecturally distinct systems that serve different purposes. Understanding their difference is critical to understanding the system.

![Reflection vs Reflexion](images/15-reflection-vs-reflexion.svg)

### Side-by-Side Comparison

| Dimension          | Reflection                   | Reflexion                               |
| ------------------ | ---------------------------- | --------------------------------------- |
| **Purpose**        | Improve the current response | Learn from failures for future queries  |
| **Scope**          | Single request               | Cross-session, cross-query              |
| **Persistence**    | None (in-memory only)        | Permanent (TimescaleDB episodic memory) |
| **Trigger**        | Always runs (if enabled)     | Only when `original_score < 0.7`        |
| **Mechanism**      | Evaluate, then refine loop   | Self-reflect, then store insight        |
| **Output**         | Refined response text        | `reflexion_insight` memory record       |
| **Read Path**      | N/A                          | Injected into future ReACT prompts      |
| **Academic Basis** | Standard self-evaluation     | Shinn et al., 2023 — Reflexion paper    |
| **Config Flag**    | `REFLECTION_ENABLED`         | `REFLEXION_ENABLED`                     |

### Reflection Deep Dive — The Post-Response Quality Gate

Reflection is the last chance to fix a response before the user sees it. It sits between Response Framing (Stage 8) and Output Guardrails (Stage 10). The system asks a separate LLM call _"Is this response actually good?"_ — and if not, a second LLM call improves it.

#### The Two LLM Roles

Reflection uses two distinct LLM calls, each with a different persona:

![The Two LLM Roles](images/16-two-llm-roles.svg)

#### The Five Quality Criteria

The evaluator scores the response on five dimensions:

| Criterion         | What It Measures                                            | Low Score Example                                             |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| **completeness**  | Does the response fully address the user's question?        | User asked to compare two products but only one was discussed |
| **accuracy**      | Is the information factually correct based on tool results? | Response says warranty is 12 months but tool returned 6       |
| **relevance**     | Is the response focused on what the user asked?             | Response includes unsolicited product recommendations         |
| **clarity**       | Is the response clear and easy to understand?               | Response is disorganized, mixing unrelated facts              |
| **actionability** | Does the response give the user useful next steps?          | Response states facts but offers no follow-up options         |

The evaluator also returns:

- `overall_score` — aggregate quality (0.0 to 1.0)
- `issues[]` — specific problems found (e.g., _"Only checked one product's warranty"_)
- `suggestions[]` — specific improvements (e.g., _"Include RoboCleaner warranty data from tool observations"_)
- `needs_refinement` — boolean, whether the evaluator thinks improvement is possible

#### The Loop Control Flow

`_run_reflection_loop()` orchestrates the evaluate-refine cycle with four exit conditions:

![Reflection Loop Control Flow](images/17-reflection-loop-control.svg)

#### Concrete Example: A Poor Response Gets Fixed

Query: _"Compare the warranty of UltraWasher 8262 with RoboCleaner 3000"_

The ReACT loop only checked one product and produced a partial answer. After framing, the response is: _"The UltraWasher 8262 has a 6-month warranty."_ (missing the RoboCleaner comparison entirely).

![Reflection Concrete Example](images/18-reflection-example.svg)

The key mechanism: the refiner LLM receives the **raw tool observations** from the ReACT loop. Even though the ReACT answer only used UltraWasher data, the tool observations contain RoboCleaner results too — the refiner uses this data to produce the complete comparison.

#### Exit Conditions Summary

| Exit Condition             | When It Fires                                    | What Happens                                       |
| -------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| **Score passes threshold** | `overall_score >= 0.75`                          | Current response (original or refined) is used     |
| **Evaluator says no**      | `needs_refinement == false` even if score is low | Response accepted as-is, no refine attempt         |
| **Max iterations reached** | 2 evaluate-refine cycles completed               | Last refined version is used                       |
| **Refine parse failure**   | Refiner LLM returns invalid JSON                 | Previous version is kept, loop breaks              |
| **Evaluate parse failure** | Evaluator LLM returns invalid JSON               | Score defaults to 1.0, response passes immediately |

Every failure mode defaults to **pass-through** — the original response is sent unchanged. Reflection can only improve a response, never block delivery.

#### Client Events During Reflection

| Event                   | When                                     | Payload                                               |
| ----------------------- | ---------------------------------------- | ----------------------------------------------------- |
| `reflection_evaluating` | Before each evaluate call                | `{iteration, step: "Evaluating response quality..."}` |
| `reflection_critique`   | After each evaluate call                 | `{iteration, score, issues[], suggestions[]}`         |
| `reflection_refining`   | Before each refine call (only if needed) | `{iteration, step: "Refining response..."}`           |

In the UI this appears as:

```
 Reflection  Evaluating response quality... Score: 0.50/1.0
             Issues: "Only one product discussed", "Missing RoboCleaner data"
 Reflection  Refining response...
 Reflection  Evaluating response quality... Score: 0.89/1.0
```

When the response scores well on the first evaluation (score >= 0.75), the user only sees a single evaluation line — no refining step appears.

#### Connection to Reflexion (Stage 11)

Reflection produces two values that Reflexion consumes downstream:

| Value             | Source                                          | Used By Reflexion                                    |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `original_score`  | First evaluation score (before any refinement)  | Gate: if `< 0.7`, store a persistent insight         |
| `last_evaluation` | The evaluation dict with issues and suggestions | Passed to LLM self-reflection for insight generation |

Reflection is the **scoring mechanism** that determines whether Reflexion fires. Without Reflection, Reflexion would have no quality signal to act on.

### Reflexion Lifecycle — Write and Read Paths

![Reflexion Lifecycle](images/19-reflexion-lifecycle.svg)

### Reflexion Deep Dive — Persistent Cross-Session Learning

Reflexion implements the academic Reflexion pattern (Shinn et al., 2023): an agent that reflects on failures, generates verbal reinforcement, stores it in persistent memory, and retrieves it on future similar queries — improving performance without retraining the model.

#### The Two Independent Code Paths

The write path and read path are completely decoupled. They run at different times and don't depend on each other within a single request.

![Reflexion Write and Read Paths](images/20-reflexion-write-read-paths.svg)

#### Write Path — Generating and Storing Insights

The write path runs at Stage 11, **after** the response has already been streamed to the user. It is entirely non-blocking — failures are logged and swallowed.

**Gate check**: The `original_score` from Reflection's first evaluation (before any refinement) must be below `REFLEXION_INSIGHT_THRESHOLD` (default 0.7). If the agent produced a good response on the first try, there is nothing to learn.

**LLM self-reflection call**: A dedicated LLM prompt analyzes the failed interaction:

```
Analyze this interaction to extract a reusable learning.

User query: What warranty does ProductX have?
Intent: warranty_question
Tools used: warranty_check
Original response quality score: 0.45
Issues found: ["Product not found in database", "Response was generic"]
Refined response quality score: 0.72

Generate a concise learning that can help handle similar queries better.
```

The LLM returns a structured insight:

```json
{
  "query_pattern": "warranty lookup for specific product",
  "failure_reason": "Used warranty_check directly with the user's query
                     text instead of finding the exact product name first",
  "suggested_improvement": "Always use product_search first to get the
                            exact product name, then call warranty_check
                            with the matched name",
  "key_topics": ["warranty", "product_search", "ProductX"]
}
```

**Storage to TimescaleDB**: The insight is stored as an immutable episodic memory:

![Insight Record in TimescaleDB](images/21-insight-record.svg)

| Field                     | Purpose                                  | Used on Read Path?                       |
| ------------------------- | ---------------------------------------- | ---------------------------------------- |
| `summary`                 | The actionable improvement text          | **Yes** — this is what the LLM sees      |
| `key_topics`              | Topic tags for relevance matching        | **Yes** — matched against query words    |
| `metadata.intent`         | Intent tag for relevance matching        | **Yes** — matched against current intent |
| `metadata.failure_reason` | What went wrong (for auditing)           | No                                       |
| `metadata.original_score` | How bad the failure was (for auditing)   | No                                       |
| `metadata.query_pattern`  | General query description (for auditing) | No                                       |
| `resolution_status`       | Whether refinement fixed it              | No                                       |

Only `summary`, `key_topics`, and `metadata.intent` participate in the read path. Everything else is audit metadata.

#### Read Path — Retrieving and Injecting Insights

The read path runs at the very start of the ReACT loop, before the first LLM call. It is completely silent — no client events are emitted.

**Fetch**: Retrieves up to 5 recent `reflexion_insight` memories for this customer from TimescaleDB.

**Relevance filtering**: Not all past insights apply to the current query. Three matching strategies are tried in order:

![Relevance Filtering](images/22-relevance-filtering.svg)

| Strategy          | How It Works                                                  | Example                                                                                           |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Intent match**  | Stored `metadata.intent` equals current query intent          | Stored insight has `intent=warranty_question`, current query is also `warranty_question`          |
| **Topic overlap** | Words in the current query intersect with stored `key_topics` | Query _"what about warranty"_ contains _"warranty"_, stored insight has `key_topics=["warranty"]` |
| **Fallback**      | No match found, use most recent insights regardless           | New intent never seen before, but recent learnings may still apply                                |

**Prompt injection**: The top 3 matching insights are formatted and injected into the `{reflexion_context}` placeholder in `REACT_SYSTEM_PROMPT`:

```
Learnings from past interactions (use these to improve your response):
- Always use product_search first to get the exact product name,
  then call warranty_check with the matched name
- When comparing warranties, check all products before generating the answer
- Include price context when answering warranty questions
```

The LLM sees these learnings before its first Thought. They influence tool selection, reasoning order, and response completeness.

#### Concrete Multi-Session Example

The following traces Reflexion across two sessions — the first where the agent fails and stores a learning, and the second where it retrieves and applies that learning.

![Multi-Session Reflexion Example](images/23-multi-session-example.svg)

The agent made a fundamentally different decision in Session 2. Without the reflexion insight, the LLM would likely call `warranty_check("ProductY")` directly — the same pattern that failed in Session 1. With the insight injected into its prompt, it called `product_search` first to resolve the exact product name.

#### Fail-Safe Design

Every failure in the reflexion system defaults to **pass-through** — it never blocks the response or the ReACT loop:

| Failure                                         | What Happens                                                      | Impact on User                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Insight generation LLM returns invalid JSON** | `_generate_reflexion_insight()` returns `None`, no insight stored | None — response already delivered                                  |
| **Memory service down during write**            | Exception caught, logged as `reflexion_insight_store_failed`      | None — response already delivered                                  |
| **Memory service down during read**             | Exception caught, returns empty string                            | ReACT loop runs without insights (normal behavior for new systems) |
| **`REFLEXION_ENABLED` is false**                | Both paths return immediately                                     | No learning, no injection                                          |
| **No `customer_id` provided**                   | Read path returns empty string                                    | Insights are per-customer; anonymous sessions get no reflexion     |
| **No past insights exist**                      | Read path returns empty string                                    | ReACT loop runs normally, as if reflexion is not installed         |

### Why Reflexion May Appear Invisible

Reflexion is a **background learning system**. It does not produce visible output during normal operation:

- **Write side**: Only triggers when `original_score < 0.7`. If the agent consistently scores well, no `reflexion_learning` event is ever emitted and no insight is ever stored.
- **Read side**: `_get_reflexion_insights()` runs silently before every ReACT loop. If no past insights exist (new system, no prior failures), it returns an empty string. There is no client event for "loaded 0 insights."

The only client-visible signal is the `reflexion_learning` event, which only appears when the system is actively learning from a poor interaction.

---

## 9. Recommendation Engine

The recommendation service generates contextual, memory-aware suggestions at two points: session start and after each response. The engine uses **focus-anchored intent strategies** — anchoring suggestions to the product/brand the user is currently discussing, combined with cross-user intelligence and episodic memory deduplication.

### 4-Tier Cold Start Strategy (GetStartRecommendations)

When a session begins, the system cascades through four tiers to generate up to 5 suggestions:

![4-Tier Cold Start Strategy](images/24-cold-start-strategy.svg)

| Tier | Name | Condition | Strategy |
|------|------|-----------|----------|
| **1C** | Returning User Override | Customer has episodic memories | 1 personal suggestion continuing last topic + fill from Tier 1A/1B |
| **1A** | Cross-User Popular Products | Platform has conversation data | Aggregate product entities across distinct customers, rank by popularity |
| **1B** | Premium Showcase | No/few popular products | Most expensive product per distinct brand, up to 3 brands |
| **Generic** | Catalog-Aware Defaults | All tiers above empty | Real brand names and price ranges from the product catalog |

**Tier 1A** counts by **product entity** (not raw query text) — "Tell me about RoboCleaner 3120" and "What's the warranty on RoboCleaner 3120?" count as one mention, not two. Each customer counts once per product.

**Tier 1B** uses `DISTINCT ON (brand)` to pick the most expensive product per brand, ensuring brand diversity. Falls back to the in-memory catalog summary when the DB is unavailable.

### Follow-Up Recommendation Pipeline (GetFollowUpRecommendations)

After every response, the system generates exactly 3 focus-anchored follow-up suggestions:

![Follow-Up Recommendation Pipeline](images/25-follow-up-pipeline.svg)

#### Step 1: Extract Current Focus

`_extract_current_focus(last_query, last_response, catalog)` derives the user's **current product** and **current brand** from the last exchange only (not the full session). It scans the response first (stronger signal — the response names the product the agent just discussed), then the query. At the same position, longer product names win (e.g., "PowerDrill 5641" beats "PowerDrill 5"). Falls back to brand-only if no full product name is found.

#### Step 2: Intent-Aware Suggestion Strategy (Slots 1-2)

The `INTENT_STRATEGY` dict maps the current intent to two suggestion templates anchored to `{current_product}`:

| Last Intent | Slot 1 | Slot 2 |
|---|---|---|
| `product_inquiry` | Warranty on {current_product} | Cost of {current_product} |
| `price_check` | Warranty on {current_product} | Compare with alternatives in similar price range |
| `warranty_question` | Price of {current_product} | Compare warranty across brands |
| `comparison` | Features of {current_product} | Price of {current_product} |
| `session_query` | Return to {current_product} | Explore new category |
| `follow_up` | Resolves to **previous** non-follow_up intent's strategy |
| *(unknown)* | Falls back to `DEFAULT_STRATEGY` |

#### Step 3: Cross-User Intelligence (Slot 3)

`_get_cooccurring_products(current_product, catalog)` finds products that appear in the same sessions as the current product across all customers — "Users who asked about X also explored Y." If no co-occurring products are found, `_find_price_alternative()` picks a product from a different brand within +/-20% of the current product's price.

#### Step 4: Dedup + Pad

Suggestions are deduplicated, filtered against the user's episodic memory (no repeat suggestions for products they've already extensively explored), and the last query echo is removed. If fewer than 3 suggestions remain, `_catalog_aware_generics()` pads with brand-aware fallbacks.

---

## 10. Data Storage Architecture

### Memory Layers

Memory is organized into five layers across three storage engines. The Memory Service provides a unified gRPC interface, routing reads and writes to the appropriate backend.

![Memory Layers](images/26-memory-layers.svg)

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Short-Term** | Redis hash per session, TTL 30min sliding | Current conversation turns, ReACT state |
| **Episodic** | TimescaleDB hypertable (7-day chunks, immutable) | Session summaries, reflexion insights, evaluation records |
| **Audit Trail** | TimescaleDB hypertable (1-day chunks, 90-day retention) | Complete event log for session replay and compliance |
| **Semantic** | PostgreSQL + pgvector (IVFFlat index) | Product embeddings for similarity search |
| **Procedural / Domain** | PostgreSQL | Tool definitions, product catalog, user data |

### Storage Schema

![Storage Schema](images/27-storage-schema.svg)

### Storage Role per Feature

| Feature              | Read From                                    | Write To           |
| -------------------- | -------------------------------------------- | ------------------ |
| Session state        | Redis (cache)                                | PostgreSQL + Redis |
| Conversation history | Redis (cache), PostgreSQL (fallback)         | PostgreSQL + Redis |
| Product catalog      | PostgreSQL (cached in-memory, 5min TTL)      | Seed data only     |
| Reflexion insights   | TimescaleDB (`event_type=reflexion_insight`) | TimescaleDB        |
| Evaluation records   | TimescaleDB                                  | TimescaleDB        |
| Audit trail          | TimescaleDB                                  | TimescaleDB        |
| Recommendations      | PostgreSQL (conversation_turns, products)    | N/A (stateless)    |

### Data Models

#### PostgreSQL (Relational + pgvector)

![PostgreSQL ER Diagram](images/28-postgresql-er-diagram.svg)

#### TimescaleDB (Immutable Hypertables)

![TimescaleDB ER Diagram](images/29-timescaledb-er-diagram.svg)

Both TimescaleDB tables are **append-only** — UPDATE and DELETE are blocked by database triggers. `episodic_memories` uses 7-day chunks; `session_audit_trail` uses 1-day chunks with a 90-day retention policy. `daily_session_stats` is a continuous aggregate materialised view, auto-refreshed hourly.

---

## 11. End-to-End Query Walkthrough

### Example: "Compare the warranty of UltraWasher 8262 with RoboCleaner 3000"

This walkthrough traces every stage, event, and service call for a real query.

![End-to-End Query Walkthrough](images/30-end-to-end-walkthrough.svg)

### What the User Sees in the UI

Based on the events emitted during this flow:

```
 Planning    "Looking up warranty for both products, then comparing"
             Steps: 1. Check UltraWasher warranty  2. Check RoboCleaner warranty  3. Compare

 Thinking    Iteration 1: "I need to check UltraWasher 8262's warranty details"
             Tool: warranty_check -> 24 months from Jan 2024

 Thinking    Iteration 2: "Now checking RoboCleaner 3000's warranty"
             Tool: warranty_check -> 36 months from Feb 2024

 Thinking    Iteration 3: "I have both warranties, ready to compare"

 Reflection  Evaluating response quality... Score: 0.88/1.0

 Response    "The RoboCleaner 3000 offers a longer warranty at 36 months
              (valid through Feb 2027) compared to the UltraWasher 8262's
              24-month warranty (valid through Jan 2026). The RoboCleaner
              gives you an additional 12 months of coverage."

              Confidence: 91%  |  Sources: UltraWasher 8262, RoboCleaner 3000

 Suggestions  "What's the warranty on RoboCleaner 3000?"
              "How much does the UltraWasher 8262 cost?"
              "Users who asked about RoboCleaner 3000 also explored PowerDrill 5641"
```

### What Happens When Quality is Poor

If the same query produced a poor response (e.g., the agent only checked one product):

```
 Reflection  Evaluating response quality... Score: 0.52/1.0
             Issues: "Only checked one product's warranty"
 Reflection  Refining response...
             Refined score: 0.74/1.0

 Reflexion   Learning from this interaction...
             Stored insight: "For warranty comparisons, always check
             all products before generating the comparison answer"

             (This insight will be injected into the system prompt
              the next time a similar warranty comparison query arrives)
```

---

## 12. Deployment Architecture

### Container Topology

![Container Topology](images/31-container-topology.svg)

### Service Dependencies

| Service | Depends On | Health Check |
|---------|------------|--------------|
| PostgreSQL | -- | `pg_isready` |
| TimescaleDB | -- | `pg_isready` |
| Redis | -- | `redis-cli ping` |
| Knowledge Service | PostgreSQL | gRPC health |
| LLM Service | -- | gRPC health |
| Memory Service | Redis, PostgreSQL, TimescaleDB | gRPC health |
| Tool Service | PostgreSQL, Knowledge Service | gRPC health |
| Recommendation Service | PostgreSQL, Memory Service | gRPC health |
| Agent Service | Memory, LLM, Knowledge, Tool, Recommendation | gRPC health |
| Gateway Service | Agent Service | HTTP `/health` |

---

## 13. Security & Authentication

### Authentication Flow

![Authentication Flow](images/32-authentication-flow.svg)

### JWT Token

```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "exp": 1234567890,
  "iat": 1234567890
}
```

- Algorithm: HS256 | Expiry: 24h (configurable via `JWT_EXPIRY_HOURS`) | Secret: `JWT_SECRET` env var

### TLS Topology

All inter-service gRPC communication uses TLS with a self-signed CA for Docker local development.

![TLS Topology](images/33-tls-topology.svg)

Certificate generation: `python scripts/generate_certs.py` creates `certs/` with CA + server certificates. SANs include all Docker service hostnames.

### Security Summary

| Layer | Mechanism | Details |
|-------|-----------|---------|
| User Authentication | Email + bcrypt password | `/api/login` endpoint |
| Session Authorization | JWT (HS256, 24h expiry) | Sent in WebSocket `session_start` |
| Transport Security | TLS 1.2+ (self-signed CA) | All gRPC channels |
| Input Guardrails | Regex injection detection | Blocks prompt injection attempts |
| Output Guardrails | Regex PII redaction | Prevents PII leakage in responses |
| Data Integrity | FK `sessions.customer_id` -> `users.id` | PostgreSQL referential integrity |
| Password Storage | bcrypt (cost factor 12) | `users.password_hash` |

---

## 14. Resilience Patterns

### Retry and Circuit Breaker

All inter-service gRPC calls use `tenacity` for retry logic:

```
Retry: 3 attempts, exponential backoff (1s, 2s, 4s)
Circuit Breaker: Open after 5 failures in 60s, half-open after 30s
Timeout: 30s per gRPC call
```

### Graceful Degradation

| Failure | Fallback |
|---------|----------|
| Knowledge Service down | Respond with "I can't search products right now" |
| LLM Service down | Return error to client |
| Memory Service down | Use in-memory fallback for session |
| Tool execution fails | Record observation as error, continue ReACT loop |
| Redis down | Memory Service falls back to PostgreSQL only |
| TimescaleDB down | Episodic writes logged and skipped (non-blocking) |
| Query rewrite fails | Use original query unchanged (non-blocking) |
| Query rewrite too long | Reject rewrite, use original query |
| Planning fails | Fall back to unguided single ReACT loop |
| Reflection parse fails | Skip refinement, use original response |
| Reflexion store fails | Log warning, continue (non-blocking) |
| Evaluation store fails | Log warning, continue (non-blocking) |
| Recommendation co-occurrence DB fails | Fall back to price alternative, then catalog generics |
| Recommendation premium showcase DB fails | Fall back to in-memory catalog summary |

---

## 15. Configuration Reference

All configuration is centralized in `shared/config.py` and driven by environment variables with sensible defaults. Every feature can be disabled independently.

### API Keys & Models

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | -- (required) | Anthropic Claude API key |
| `VOYAGE_API_KEY` | -- (required) | Voyage AI embedding API key |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model identifier |
| `EMBEDDING_MODEL` | `voyage-3` | Voyage embedding model |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding vector dimensions |

### Agent Pipeline

| Setting                       | Default | Description                                          |
| ----------------------------- | ------- | ---------------------------------------------------- |
| `REACT_MAX_ITERATIONS`        | `8`     | Maximum ReACT reasoning steps per query              |
| `REACT_TIMEOUT_SECONDS`       | `120`   | Total pipeline timeout in seconds                    |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.8`   | Below this, ask for clarification                    |
| `DOMAIN_RELEVANCE_THRESHOLD`  | `0.5`   | Below this, redirect as out-of-scope                 |

### Memory Context

| Setting                          | Default | Description                                        |
| -------------------------------- | ------- | -------------------------------------------------- |
| `MEMORY_CONTEXT_MAX_TURNS`       | `10`    | Max conversation turns used for structured context  |
| `MEMORY_CONTEXT_TRUNCATE_LENGTH` | `200`   | Char limit for older assistant responses in context |

### Query Rewriting

| Setting                      | Default | Description                                            |
| ---------------------------- | ------- | ------------------------------------------------------ |
| `QUERY_REWRITE_ENABLED`      | `true`  | Enable LLM-based pronoun/reference resolution          |
| `QUERY_REWRITE_TEMPERATURE`  | `0.1`   | LLM temperature for query rewrite (low = deterministic)|
| `QUERY_REWRITE_MAX_TOKENS`   | `128`   | Max tokens for query rewrite response                  |

### Reflection (Stage 9)

| Setting                        | Default | Description                               |
| ------------------------------ | ------- | ----------------------------------------- |
| `REFLECTION_ENABLED`           | `true`  | Enable post-response quality evaluation   |
| `REFLECTION_MAX_ITERATIONS`    | `2`     | Max evaluate-refine cycles                |
| `REFLECTION_QUALITY_THRESHOLD` | `0.75`  | Score above which refinement stops        |
| `REFLECTION_TEMPERATURE`       | `0.2`   | LLM temperature for evaluation/refinement |
| `REFLECTION_MAX_TOKENS`        | `512`   | Max tokens for evaluation/refinement      |

### Reflexion (Stage 11)

| Setting                            | Default | Description                              |
| ---------------------------------- | ------- | ---------------------------------------- |
| `REFLEXION_ENABLED`                | `true`  | Enable persistent cross-session learning |
| `REFLEXION_INSIGHT_THRESHOLD`      | `0.7`   | Store insight if score below this        |
| `REFLEXION_MAX_INSIGHTS_PER_QUERY` | `3`     | Max insights injected per query          |

### Planning & Multi-Agent

| Setting                  | Default | Description                      |
| ------------------------ | ------- | -------------------------------- |
| `PLANNING_ENABLED`       | `true`  | Enable query decomposition       |
| `PLANNING_TEMPERATURE`   | `0.2`   | LLM temperature for planning     |
| `MULTI_AGENT_ENABLED`    | `true`  | Enable specialist agent dispatch |
| `MULTI_AGENT_MAX_AGENTS` | `3`     | Max specialists per query        |

### Guardrails

| Setting                       | Default | Description                       |
| ----------------------------- | ------- | --------------------------------- |
| `GUARDRAILS_ENABLED`          | `true`  | Enable input/output safety checks |
| `GUARDRAILS_MAX_QUERY_LENGTH` | `2000`  | Max characters per query          |

### LLM Defaults

| Setting               | Default                    | Description                           |
| --------------------- | -------------------------- | ------------------------------------- |
| `LLM_MODEL`           | `claude-sonnet-4-20250514` | Claude model for all LLM calls        |
| `DEFAULT_TEMPERATURE` | `0.3`                      | Default temperature (ReACT reasoning) |
| `DEFAULT_MAX_TOKENS`  | `1024`                     | Default max tokens                    |
| `INTENT_TEMPERATURE`  | `0.1`                      | Temperature for intent classification |
| `INTENT_MAX_TOKENS`   | `256`                      | Max tokens for intent classification  |

### Storage

| Setting               | Default                                              | Description                    |
| --------------------- | ---------------------------------------------------- | ------------------------------ |
| `DATABASE_URL`        | `postgresql://piper:piper@postgres:5432/piper`       | PostgreSQL connection          |
| `TIMESCALEDB_URL`     | `postgresql://piper:piper@timescaledb:5432/piper_ts` | TimescaleDB connection         |
| `REDIS_URL`           | `redis://redis:6379/0`                               | Redis connection               |
| `SESSION_TTL_SECONDS` | `1800`                                               | Session cache TTL (30 minutes) |

### Security & Auth

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | `piper-dev-secret-change-in-prod` | JWT signing secret |
| `JWT_EXPIRY_HOURS` | `24` | Token expiry duration |
| `TLS_CA_CERT` | `/app/certs/ca.pem` | CA certificate path |
| `TLS_SERVER_CERT` | `/app/certs/server.pem` | Server certificate path |
| `TLS_SERVER_KEY` | `/app/certs/server-key.pem` | Server private key path |

### Service Addresses

| Variable | Default |
|---|---|
| `AGENT_SERVICE_ADDR` | `agent_service:50054` |
| `MEMORY_SERVICE_ADDR` | `memory_service:50055` |
| `LLM_SERVICE_ADDR` | `llm_service:50053` |
| `KNOWLEDGE_SERVICE_ADDR` | `knowledge_service:50052` |
| `TOOL_SERVICE_ADDR` | `tool_service:50056` |
| `RECOMMENDATION_SERVICE_ADDR` | `recommendation_service:50057` |

When all pipeline features are disabled (`PLANNING_ENABLED=false`, `MULTI_AGENT_ENABLED=false`, `GUARDRAILS_ENABLED=false`, `REFLECTION_ENABLED=false`, `REFLEXION_ENABLED=false`, `EVALUATION_STORAGE_ENABLED=false`, `QUERY_REWRITE_ENABLED=false`), the system operates as a basic ReACT agent with intent classification and tool use.

---

## Event Reference

All events streamed to the client via WebSocket during query processing:

| Event Type              | Stage | Payload                                      | Description                      |
| ----------------------- | ----- | -------------------------------------------- | -------------------------------- |
| `processing_started`    | 0     | `{}`                                         | Query processing began           |
| `guardrail_blocked`     | 1     | `{reason, type}`                             | Input blocked by safety filter   |
| `clarification`         | 4.B   | `{message, options, allow_freetext}`         | Asking user for more context     |
| `agent_planning`        | 5     | `{steps[], multi_agent}`                     | Plan decomposition result        |
| `agent_started`         | 6     | `{agent_type, description}`                  | Specialist agent launched        |
| `agent_thinking`        | 6     | `{iteration, thought, action, has_answer}`   | ReACT reasoning step             |
| `tool_validation_error` | 6     | `{tool, error}`                              | Tool parameter validation failed |
| `agent_complete`        | 6     | `{agent_type, tools_used[]}`                 | Specialist agent finished        |
| `reflection_evaluating` | 9     | `{iteration}`                                | Evaluating response quality      |
| `reflection_critique`   | 9     | `{score, issues[], needs_refinement}`        | Quality evaluation result        |
| `reflection_refining`   | 9     | `{iteration}`                                | Improving response               |
| `guardrail_sanitized`   | 10    | `{redactions[]}`                             | PII redacted from output         |
| `reflexion_learning`    | 11    | `{message}`                                  | Learning from poor interaction   |
| `token`                 | 13    | `{token}`                                    | Streamed response token          |
| `response_complete`     | 13    | `{text, confidence, sources, suggestions[]}` | Final response with metadata     |
| `error`                 | Any   | `{message, code}`                            | Error occurred                   |

---

## Tool Reference

Four domain tools are available to the ReACT engine:

| Tool              | Parameters                                              | Returns                                        | Use Case                        |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------- | ------------------------------- |
| `product_search`  | `query` (str), `top_k` (int)                            | Semantic search results with similarity scores | Finding products by description |
| `price_lookup`    | `product_name` (str) OR `min_price`/`max_price` (float) | Price data with warranty info                  | Price queries, budget filtering |
| `warranty_check`  | `product_name` (str)                                    | Warranty months, manufacturing date, price     | Warranty coverage questions     |
| `product_compare` | `product_names` (list of str)                           | Side-by-side product comparison                | Multi-product comparisons       |

`product_search` uses **Voyage AI embeddings** (`voyage-3`, 1024 dimensions) for semantic similarity, while the other three tools query PostgreSQL directly.

---

_Piper AI Agent_
