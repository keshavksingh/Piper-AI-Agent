# Piper AI Agent — System Architecture

> A production-grade, multi-agent customer support system built on gRPC microservices, ReACT reasoning, and persistent learning through Reflexion.

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [Service Topology](#2-service-topology)
3. [The 12-Stage Query Pipeline](#3-the-12-stage-query-pipeline)
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

```mermaid
graph TB
    subgraph Client Layer
        UI["Web UI<br/>(React + WebSocket)"]
    end

    subgraph Gateway Layer
        GW["Gateway Server<br/>WebSocket + JWT Auth<br/>Rate Limiting"]
    end

    subgraph Orchestration Layer
        AS["Agent Service<br/>12-Stage Pipeline<br/>ReACT + Multi-Agent"]
    end

    subgraph Intelligence Layer
        LLM["LLM Service<br/>Claude claude-sonnet-4-20250514<br/>Prompt Routing"]
        KS["Knowledge Service<br/>Voyage Embeddings<br/>Semantic Search"]
    end

    subgraph Execution Layer
        TS["Tool Service<br/>4 Domain Tools<br/>Param Validation"]
        RS["Recommendation Service<br/>3-Tier Cold Start<br/>Gap Analysis Engine"]
    end

    subgraph Memory Layer
        MS["Memory Service<br/>Sessions + History<br/>Episodic Memories"]
    end

    subgraph Storage Layer
        PG[("PostgreSQL<br/>Sessions, Products<br/>Conversations")]
        TS_DB[("TimescaleDB<br/>Episodic Memories<br/>Audit Trail")]
        RD[("Redis<br/>Session Cache<br/>TTL: 1800s")]
    end

    UI <-->|"WebSocket"| GW
    GW <-->|"gRPC Stream"| AS
    AS <-->|"gRPC"| LLM
    AS <-->|"gRPC"| TS
    AS <-->|"gRPC"| MS
    AS <-->|"gRPC"| RS
    TS <-->|"gRPC"| KS
    MS <--> PG
    MS <--> TS_DB
    MS <--> RD

    style UI fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style GW fill:#F5A623,stroke:#C47D0E,color:#fff
    style AS fill:#D0021B,stroke:#9B0016,color:#fff
    style LLM fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style KS fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style TS fill:#50C878,stroke:#3A9A5C,color:#fff
    style RS fill:#50C878,stroke:#3A9A5C,color:#fff
    style MS fill:#FF6B6B,stroke:#CC5555,color:#fff
    style PG fill:#336791,stroke:#264E6D,color:#fff
    style TS_DB fill:#336791,stroke:#264E6D,color:#fff
    style RD fill:#DC382D,stroke:#A82B23,color:#fff
```

---

## 2. Service Topology

Each service runs as an independent gRPC server. All inter-service communication uses Protocol Buffers.

```mermaid
graph LR
    subgraph "Proto Contracts"
        P1["agent_service.proto<br/>ProcessQuery (stream)<br/>SubmitClarification (stream)"]
        P2["memory_service.proto<br/>CreateSession, GetSession<br/>AddTurn, GetHistory<br/>StoreEpisodic, GetEpisodic"]
        P3["tool_service.proto<br/>ListTools<br/>ExecuteTool"]
        P4["llm_service.proto<br/>GenerateAnswer"]
        P5["knowledge_service.proto<br/>SemanticSearch"]
        P6["recommendation_service.proto<br/>GetStartRecommendations<br/>GetFollowUpRecommendations"]
    end

    style P1 fill:#D0021B,stroke:#9B0016,color:#fff
    style P2 fill:#FF6B6B,stroke:#CC5555,color:#fff
    style P3 fill:#50C878,stroke:#3A9A5C,color:#fff
    style P4 fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style P5 fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style P6 fill:#50C878,stroke:#3A9A5C,color:#fff
```

| Service                    | Address             | Role                                                      |
| -------------------------- | ------------------- | --------------------------------------------------------- |
| **Gateway Server**         | `:8765` (WebSocket) | Client-facing entry point, JWT auth, rate limiting        |
| **Agent Service**          | `:50054` (gRPC)     | Core orchestrator — 12-stage pipeline, ReACT, multi-agent |
| **Memory Service**         | `:50055` (gRPC)     | Session state, conversation history, episodic memories    |
| **Tool Service**           | `:50056` (gRPC)     | Domain tool execution with schema validation              |
| **LLM Service**            | `:50053` (gRPC)     | Claude API wrapper with temperature/token routing         |
| **Knowledge Service**      | `:50052` (gRPC)     | Voyage AI embeddings + semantic product search            |
| **Recommendation Service** | `:50057` (gRPC)     | Context-aware suggestion generation                       |

### Request Flow Summary

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as Gateway
    participant AG as Agent
    participant MEM as Memory
    participant LLM as LLM
    participant REC as Recommendation

    C->>GW: POST /api/login {email, password}
    GW->>GW: Verify bcrypt password
    GW-->>C: {token: "JWT..."}
    C->>GW: WebSocket connect
    C->>GW: {type: "session_start", token: "JWT..."}
    GW->>GW: Verify JWT, extract user_id
    GW->>MEM: Load/create session (user_id)
    GW->>REC: Get start recommendations
    GW-->>C: {"type":"recommendations", ...}

    C->>GW: {"type":"user_message", "text":"..."}
    GW->>AG: ProcessQuery(session_id, query)
    Note over AG: Pipeline stages 0-12 (see Section 3)
    AG-->>GW: Stream events (tokens, indicators, response_complete)
    GW-->>C: {"type":"token", ...}
    GW-->>C: {"type":"response_complete", ...}
```

---

## 3. The 12-Stage Query Pipeline

Every user query flows through a deterministic 12-stage pipeline inside `ProcessQuery()`. Each stage is independently feature-flagged.

```mermaid
flowchart TD
    START(["User Query Received"])

    S0["Stage 0: Session Context<br/>Touch session, load last 10 turns"]
    S1{"Stage 1: Input Guardrails<br/>Length, injection, PII checks"}
    S1_BLOCK(["BLOCKED<br/>guardrail_blocked event"])

    S2["Stage 2: Intent Classification<br/>LLM classifies intent + confidence<br/>temp=0.1, max_tokens=256"]
    S3{"Stage 3: Clarification<br/>confidence below 0.8?"}
    S3_ASK(["ASK USER<br/>clarification event"])

    S4["Stage 4: Planning Layer<br/>LLM decomposes into sub-goals<br/>Decides single vs multi-agent<br/>temp=0.2, max_tokens=512"]
    S4_EVENT["Emit: agent_planning"]

    S5{"Stage 5: Routing<br/>Simple intent?<br/>Multi-agent?"}
    S5_SIMPLE["Direct LLM Response<br/>No tools needed"]
    S5_MULTI["Multi-Agent Loop<br/>Up to 3 specialists"]
    S5_REACT["Single ReACT Loop<br/>Up to 8 iterations"]

    S7["Stage 7: Response Framing<br/>LLM polishes answer<br/>Adds confidence + sources"]
    S8["Stage 8: Reflection<br/>Evaluate quality (5 criteria)<br/>Refine if score below 0.75<br/>Max 2 refinement cycles"]
    S9["Stage 9: Output Guardrails<br/>PII redaction"]
    S10{"Stage 10: Reflexion<br/>Original score below 0.7?"}
    S10_YES["Generate + Store Insight<br/>to TimescaleDB episodic memory<br/>Emit: reflexion_learning"]
    S10_NO["Skip storage"]
    S11["Stage 11: Evaluation Storage<br/>Metrics to TimescaleDB"]
    S12["Stage 12: Recommendations<br/>+ Token Streaming<br/>Emit: response_complete"]

    START --> S0 --> S1
    S1 -->|"Safe"| S2
    S1 -->|"Blocked"| S1_BLOCK
    S2 --> S3
    S3 -->|"Needs clarification"| S3_ASK
    S3 -->|"Confident"| S4
    S4 --> S4_EVENT --> S5
    S5 -->|"general_question<br/>or out_of_scope"| S5_SIMPLE
    S5 -->|"needs_multi_agent=true"| S5_MULTI
    S5 -->|"Single agent"| S5_REACT
    S5_SIMPLE --> S12
    S5_MULTI --> S7
    S5_REACT --> S7
    S7 --> S8 --> S9 --> S10
    S10 -->|"Score below 0.7"| S10_YES --> S11
    S10 -->|"Score at or above 0.7"| S10_NO --> S11
    S11 --> S12

    style START fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style S0 fill:#6C757D,stroke:#495057,color:#fff
    style S1 fill:#DC3545,stroke:#A71D2A,color:#fff
    style S1_BLOCK fill:#DC3545,stroke:#A71D2A,color:#fff
    style S2 fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style S3 fill:#FFC107,stroke:#CC9A06,color:#000
    style S3_ASK fill:#FFC107,stroke:#CC9A06,color:#000
    style S4 fill:#17A2B8,stroke:#117A8B,color:#fff
    style S4_EVENT fill:#17A2B8,stroke:#117A8B,color:#fff
    style S5 fill:#6F42C1,stroke:#59359A,color:#fff
    style S5_SIMPLE fill:#28A745,stroke:#1E7E34,color:#fff
    style S5_MULTI fill:#E83E8C,stroke:#B5305F,color:#fff
    style S5_REACT fill:#FD7E14,stroke:#CA6510,color:#fff
    style S7 fill:#20C997,stroke:#199B76,color:#fff
    style S8 fill:#6610F2,stroke:#510EC0,color:#fff
    style S9 fill:#DC3545,stroke:#A71D2A,color:#fff
    style S10 fill:#E65100,stroke:#BF4400,color:#fff
    style S10_YES fill:#E65100,stroke:#BF4400,color:#fff
    style S10_NO fill:#6C757D,stroke:#495057,color:#fff
    style S11 fill:#6C757D,stroke:#495057,color:#fff
    style S12 fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

---

## 4. Detailed Stage Walkthrough

### Stage 0 — Session Context

```
Session Touch (Redis TTL refresh)
    +-- Load last 10 conversation turns from PostgreSQL
    +-- Build memory_context string: "User: ... \n Assistant: ..."
    +-- Store current user query as new conversation turn
```

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

### Stage 2 — Intent Classification

The LLM classifies the user's query into one of 7 intent categories:

| Intent              | Description                      | Routing               |
| ------------------- | -------------------------------- | --------------------- |
| `product_inquiry`   | Questions about product details  | ReACT or Multi-Agent  |
| `price_check`       | Price lookups and budget queries | ReACT or Multi-Agent  |
| `comparison`        | Comparing products/brands        | ReACT or Multi-Agent  |
| `warranty_question` | Warranty coverage and claims     | ReACT or Multi-Agent  |
| `follow_up`         | Continuing previous conversation | ReACT or Multi-Agent  |
| `general_question`  | General chat, greetings          | Direct LLM (no tools) |
| `out_of_scope`      | Off-topic queries                | Polite redirect       |

### Stage 3 — Clarification Gate

If intent `confidence < 0.8` and `needs_clarification = true`, the system sends a structured clarification request with clickable options:

```json
{
  "type": "clarification",
  "message": "I want to make sure I help you correctly. Could you clarify:",
  "options": [
    {"label": "I'm looking for product recommendations", "value": "product_inquiry"},
    {"label": "I want to compare prices", "value": "price_check"},
    {"label": "I have a warranty question", "value": "warranty_question"}
  ],
  "allow_freetext": true
}
```

The user's response is merged with the original query and re-classified (loops back to Stage 2).

### Stage 4 — Planning Layer

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

### Stage 9 — Output Guardrails (PII Redaction)

The same PII regex patterns from Stage 1 are applied to the response text. Unlike input guardrails, output guardrails **redact** rather than block:

| PII Type    | Example              | Redaction          |
|-------------|----------------------|--------------------|
| Email       | `user@example.com`   | `[EMAIL REDACTED]` |
| Phone       | `(555) 123-4567`     | `[PHONE REDACTED]` |
| SSN         | `123-45-6789`        | `[SSN REDACTED]`   |
| Credit Card | `1234 5678 9012 3456`| `[CARD REDACTED]`  |

`_check_output_guardrails(response_text)` returns `(sanitized_text, was_modified, redactions)`. If PII is redacted, a `guardrail_sanitized` event is emitted.

### Stage 11 — Evaluation Storage

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
| Planning | +1 | ~1-2s |
| Multi-agent (2 specialists + synthesis) | +3-5 | ~5-10s |
| Reflection: passes (score >= 0.75) | +1 | ~1-2s |
| Reflection: 1 refinement | +2 | ~2-4s |
| Reflection: 2 refinements (max) | +4 | ~4-8s |
| Reflexion insight stored | +1 | ~1-2s |
| Output guardrails (regex) | 0 | <1ms |
| Evaluation storage (DB write) | 0 | ~50ms |

Worst case (planning + multi-agent + 2 reflection refinements + reflexion): ~15-20s additional. Well within the 120s timeout.

### Pipeline Methods Reference

All methods on `AgentServiceServicer` in `agent_service/server.py`:

| Method | Stage | Purpose |
|---|---|---|
| `ProcessQuery()` | -- | Main entry point; orchestrates all stages |
| `_check_input_guardrails(query)` | 1 | Regex PII + injection detection on input |
| `_classify_intent(query, context)` | 2 | LLM call for intent classification |
| `_build_clarification_options(intent_result)` | 3 | Build clarification option list |
| `_generate_plan(query, intent, tool_list, context)` | 4 | LLM call to decompose query into sub-goals |
| `_execute_react_step(query, memory, tools, history)` | 5 | Single ReACT iteration via LLM |
| `_validate_tool_params(tool_name, params, schema)` | 5 | Validate tool call parameters against schema |
| `_execute_tool(session_id, tool_name, params)` | 5 | Execute tool via Tool Service gRPC |
| `_validate_tool_result(tool_name, result)` | 5 | Validate tool output completeness |
| `_get_reflexion_insights(customer_id, intent, query)` | 5 | Fetch past learnings from episodic memory |
| `_run_react_loop(session_id, ...)` | 5 | Full ReACT loop orchestration |
| `_run_agent_sub_loop(agent_type, query, ...)` | 6 | Focused 4-iteration ReACT loop for specialist |
| `_run_multi_agent_loop(session_id, ...)` | 6 | Orchestrate sequential specialist agents |
| `_synthesize_multi_agent_response(results, query)` | 6 | LLM call to combine specialist outputs |
| `_frame_response(query, answer, tools, steps)` | 7 | LLM call to polish answer with metadata |
| `_evaluate_response(query, text, tools, steps, ctx)` | 8 | LLM call to score response quality |
| `_refine_response(query, text, tools, eval, obs)` | 8 | LLM call to improve response |
| `_run_reflection_loop(query, framed, tools, steps, ctx)` | 8 | Evaluate-refine loop orchestration |
| `_check_output_guardrails(response_text)` | 9 | PII redaction on output |
| `_generate_reflexion_insight(query, intent, ...)` | 10 | LLM call to produce reusable learning |
| `_maybe_store_reflexion_insight(session_id, ...)` | 10 | Store insight if quality below threshold |
| `_store_evaluation_record(session_id, ...)` | 11 | Store evaluation metrics to TimescaleDB |
| `_handle_simple_intent(session_id, ...)` | 2->9->12 | Handle general_question/out_of_scope directly |

---

## 5. ReACT Reasoning Engine

The core reasoning loop implements the **ReACT** (Reasoning + Acting) framework. The agent alternates between thinking and tool execution until it has enough information to answer.

```mermaid
flowchart TD
    ENTRY(["Enter ReACT Loop"])

    LOAD_INSIGHTS["Load Reflexion Insights<br/>from episodic memory<br/>(up to 3 past learnings)"]
    INJECT_PLAN["Inject Plan Steps<br/>into system prompt"]

    BUILD["Build System Prompt<br/>= Base Prompt<br/>+ Reflexion Context<br/>+ Plan Steps<br/>+ Tool Descriptions"]

    ITER{"Iteration i<br/>(max 8)"}
    TIMEOUT{"Elapsed > 120s?"}
    TIMEOUT_MSG["Return timeout message"]

    LLM_CALL["LLM Call: _execute_react_step<br/>temp=0.3, max_tokens=1024"]
    PARSE["Parse Output:<br/>Thought + Action or Answer"]

    HAS_ANSWER{"Has Answer?"}
    HAS_ACTION{"Has Action?"}

    VALIDATE_PARAMS{"Validate Tool Params<br/>Required fields?<br/>Correct types?"}
    INVALID_PARAMS["Observation: Parameter error<br/>(self-correction hint)"]
    EXECUTE_TOOL["Execute Tool via<br/>Tool Service gRPC"]
    VALIDATE_RESULT{"Validate Result<br/>Error keys?<br/>Empty results?"}
    ENRICH["Enrich Observation<br/>with guidance"]
    APPEND["Append Step to History<br/>(thought, action, observation)"]

    EMIT_THINK["Emit: agent_thinking<br/>(iteration, thought)"]

    FRAME["Stage 7: Frame Response<br/>LLM polishes final answer"]

    ENTRY --> LOAD_INSIGHTS --> INJECT_PLAN --> BUILD --> ITER
    ITER --> TIMEOUT
    TIMEOUT -->|"Yes"| TIMEOUT_MSG
    TIMEOUT -->|"No"| LLM_CALL
    LLM_CALL --> PARSE --> EMIT_THINK
    EMIT_THINK --> HAS_ANSWER
    HAS_ANSWER -->|"Yes"| FRAME
    HAS_ANSWER -->|"No"| HAS_ACTION
    HAS_ACTION -->|"Yes"| VALIDATE_PARAMS
    HAS_ACTION -->|"No fallback"| FRAME
    VALIDATE_PARAMS -->|"Valid"| EXECUTE_TOOL
    VALIDATE_PARAMS -->|"Invalid"| INVALID_PARAMS --> APPEND
    EXECUTE_TOOL --> VALIDATE_RESULT
    VALIDATE_RESULT -->|"Clean"| APPEND
    VALIDATE_RESULT -->|"Issues"| ENRICH --> APPEND
    APPEND --> ITER

    style ENTRY fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style LOAD_INSIGHTS fill:#E65100,stroke:#BF4400,color:#fff
    style INJECT_PLAN fill:#17A2B8,stroke:#117A8B,color:#fff
    style BUILD fill:#6C757D,stroke:#495057,color:#fff
    style LLM_CALL fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style PARSE fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style EXECUTE_TOOL fill:#50C878,stroke:#3A9A5C,color:#fff
    style VALIDATE_PARAMS fill:#DC3545,stroke:#A71D2A,color:#fff
    style VALIDATE_RESULT fill:#DC3545,stroke:#A71D2A,color:#fff
    style FRAME fill:#20C997,stroke:#199B76,color:#fff
    style EMIT_THINK fill:#FFC107,stroke:#CC9A06,color:#000
    style TIMEOUT fill:#DC3545,stroke:#A71D2A,color:#fff
    style TIMEOUT_MSG fill:#DC3545,stroke:#A71D2A,color:#fff
    style INVALID_PARAMS fill:#FFC107,stroke:#CC9A06,color:#000
    style ENRICH fill:#FFC107,stroke:#CC9A06,color:#000
```

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

```mermaid
flowchart LR
    A["Action:<br/>warranty_check"] --> B{"Parameter<br/>Validation"}
    B -->|"Missing required<br/>field"| C["Observation:<br/>Missing 'product_name'<br/>Please provide it"]
    B -->|"Wrong type"| D["Observation:<br/>Expected string<br/>got integer"]
    B -->|"Valid"| E["Execute Tool"]
    E --> F{"Result<br/>Validation"}
    F -->|"Error key found"| G["Enriched:<br/>Tool returned error<br/>Try alternative"]
    F -->|"Empty results"| H["Enriched:<br/>No results found<br/>Try broader search"]
    F -->|"Clean"| I["Raw observation<br/>passed to LLM"]

    style B fill:#DC3545,stroke:#A71D2A,color:#fff
    style F fill:#DC3545,stroke:#A71D2A,color:#fff
    style E fill:#50C878,stroke:#3A9A5C,color:#fff
    style C fill:#FFC107,stroke:#CC9A06,color:#000
    style D fill:#FFC107,stroke:#CC9A06,color:#000
    style G fill:#FFC107,stroke:#CC9A06,color:#000
    style H fill:#FFC107,stroke:#CC9A06,color:#000
    style I fill:#28A745,stroke:#1E7E34,color:#fff
```

Invalid parameters are **not blocked** — they're returned as observations so the LLM can self-correct on the next iteration.

### System Prompt Assembly

Before the loop begins, the system constructs a composite prompt from four injected sections. This prompt stays constant across all iterations — only the user prompt (with accumulating history) changes per iteration.

```mermaid
flowchart TD
    subgraph "System Prompt (constant across iterations)"
        BASE["Base Persona<br/>'You are Piper, an AI customer<br/>support agent...<br/>You must reason step-by-step<br/>using the ReACT framework.'"]
        REFL["Reflexion Context<br/>(from past poor interactions)<br/>'Learnings from past interactions:<br/>- When comparing warranties,<br/>  always check both products'"]
        PLAN["Plan Context<br/>(from Stage 4 planning)<br/>'Execution plan:<br/>1. Look up warranty (warranty_check)<br/>2. Look up price (price_lookup)'"]
        TOOLS["Tool Descriptions<br/>'product_search(query, top_k)<br/>price_lookup(product_name)<br/>warranty_check(product_name)<br/>product_compare(product_names)'"]
        RULES["Output Rules<br/>'Pattern 1: Thought + Action<br/>Pattern 2: Thought + Answer'"]
    end

    subgraph "User Prompt (changes each iteration)"
        MEM["Session Context<br/>Last 10 conversation turns"]
        QUERY["User Query<br/>'What is the warranty on<br/>UltraWasher 8262?'"]
        HIST["ReACT History<br/>(empty on iteration 1,<br/>grows each iteration)"]
    end

    BASE --> REFL --> PLAN --> TOOLS --> RULES
    MEM --> QUERY --> HIST

    style BASE fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style REFL fill:#E65100,stroke:#BF4400,color:#fff
    style PLAN fill:#17A2B8,stroke:#117A8B,color:#fff
    style TOOLS fill:#50C878,stroke:#3A9A5C,color:#fff
    style RULES fill:#6C757D,stroke:#495057,color:#fff
    style MEM fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style QUERY fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style HIST fill:#FFC107,stroke:#CC9A06,color:#000
```

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

```mermaid
sequenceDiagram
    participant RL as ReACT Loop
    participant LLM as LLM Service
    participant Val as Param Validator
    participant Tool as Tool Service
    participant DB as PostgreSQL

    Note over RL: Iteration 1 starts
    Note over RL: react_history = "No previous reasoning steps."

    RL->>LLM: System prompt + User prompt + empty history
    LLM-->>RL: Thought: User wants warranty and price.<br/>I will check warranty first.<br/>Action: warranty_check({"product_name":"UltraWasher 8262"})

    Note over RL: parse_react_output() extracts:<br/>thought, action="warranty_check",<br/>action_input={"product_name":"UltraWasher 8262"},<br/>answer=None

    RL->>Val: Validate params against schema
    Note over Val: Required field "product_name"? Present.<br/>Type is string? Yes.<br/>Result: VALID

    RL->>Tool: ExecuteTool("warranty_check", params)
    Tool->>DB: SELECT warranty_months, price,<br/>manufacturing_date FROM products<br/>WHERE product_name LIKE '%UltraWasher 8262%'
    DB-->>Tool: warranty_months=6, price=121.24,<br/>manufacturing_date=2023-09-21
    Tool-->>RL: {"results":[{"product_name":"UltraWasher 8262",<br/>"warranty_months":6,"price":121.24,<br/>"manufacturing_date":"2023-09-21"}],"count":1}

    Note over RL: Validate result: no "error" key,<br/>results list non-empty. CLEAN.

    Note over RL: Append to steps[]:<br/>{iteration:1, thought, action,<br/>action_input, observation}

    Note over RL: Iteration 2 starts
    Note over RL: build_react_history(steps) produces:<br/>"Previous reasoning:<br/>Thought 1: User wants warranty and price...<br/>Action 1: warranty_check({...})<br/>Observation 1: {results:[{warranty_months:6,<br/>price:121.24,...}]}"

    RL->>LLM: System prompt + User prompt + accumulated history
    LLM-->>RL: Thought: The warranty_check already returned<br/>the price ($121.24) and warranty (6 months).<br/>I have everything needed.<br/>Answer: The UltraWasher 8262 is priced at $121.24<br/>and comes with a 6-month warranty from Sept 2023.

    Note over RL: parse_react_output() extracts:<br/>answer="The UltraWasher 8262 is priced at..."<br/>answer is NOT None -> break loop

    Note over RL: final_answer set, loop exits.<br/>Proceed to Response Framing (Stage 7).
```

### How History Accumulates

The critical mechanism is `build_react_history()` — each iteration's thought, action, and observation are appended to a growing history string. On every subsequent LLM call, the model sees **all** previous reasoning, giving it memory within the loop.

```mermaid
flowchart TD
    subgraph "Iteration 1 — LLM sees"
        H1["react_history:<br/>'No previous reasoning steps.'"]
    end

    subgraph "Iteration 2 — LLM sees"
        H2["react_history:<br/>'Previous reasoning:<br/>Thought 1: User wants warranty and price...<br/>Action 1: warranty_check({product_name:...})<br/>Observation 1: {results:[{warranty_months:6,<br/>price:121.24}], count:1}'"]
    end

    subgraph "Iteration 3 — LLM sees"
        H3["react_history:<br/>'Previous reasoning:<br/>Thought 1: ...<br/>Action 1: warranty_check({...})<br/>Observation 1: {results:[...]}<br/>Thought 2: Now I need the competing product...<br/>Action 2: warranty_check({product_name:RoboCleaner})<br/>Observation 2: {results:[{warranty_months:36}]}'"]
    end

    H1 -->|"+ step 1 appended"| H2
    H2 -->|"+ step 2 appended"| H3

    style H1 fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style H2 fill:#17A2B8,stroke:#117A8B,color:#fff
    style H3 fill:#7B68EE,stroke:#5A4ACB,color:#fff
```

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

```mermaid
flowchart TD
    ERR1["Invalid Parameters<br/>LLM sends wrong field name<br/>or wrong type"]
    ERR2["Empty Results<br/>Tool returns zero matches"]
    ERR3["Unknown Tool<br/>LLM hallucinates a tool name"]
    ERR4["Timeout<br/>Pipeline exceeds 120 seconds"]
    ERR5["Max Iterations<br/>8 iterations with no Answer"]

    FIX1["Observation becomes:<br/>'Missing required field product_name.<br/>Please fix the parameters<br/>and try again.'<br/>LLM self-corrects next iteration"]
    FIX2["Observation enriched with:<br/>'No results found. Consider<br/>broadening your search or<br/>using different terms.'<br/>LLM tries alternative query"]
    FIX3["Observation becomes:<br/>'Unknown tool: check_inventory.<br/>Available tools: product_search,<br/>price_lookup, warranty_check,<br/>product_compare'<br/>LLM picks a valid tool"]
    FIX4["Loop breaks immediately.<br/>Returns: 'I am taking too long<br/>to process your request.<br/>Please try a simpler question.'"]
    FIX5["Compiles all observations<br/>collected so far into a<br/>best-effort answer:<br/>'Based on what I found:<br/>- warranty_check: {results...}<br/>- price_lookup: {results...}'"]

    ERR1 --> FIX1
    ERR2 --> FIX2
    ERR3 --> FIX3
    ERR4 --> FIX4
    ERR5 --> FIX5

    style ERR1 fill:#DC3545,stroke:#A71D2A,color:#fff
    style ERR2 fill:#DC3545,stroke:#A71D2A,color:#fff
    style ERR3 fill:#DC3545,stroke:#A71D2A,color:#fff
    style ERR4 fill:#DC3545,stroke:#A71D2A,color:#fff
    style ERR5 fill:#DC3545,stroke:#A71D2A,color:#fff
    style FIX1 fill:#28A745,stroke:#1E7E34,color:#fff
    style FIX2 fill:#FFC107,stroke:#CC9A06,color:#000
    style FIX3 fill:#28A745,stroke:#1E7E34,color:#fff
    style FIX4 fill:#6C757D,stroke:#495057,color:#fff
    style FIX5 fill:#6C757D,stroke:#495057,color:#fff
```

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

This framed response then continues through Reflection (Stage 8), Output Guardrails (Stage 9), Reflexion (Stage 10), and finally streams to the client.

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

```mermaid
flowchart LR
    subgraph "Product Record"
        PR["UltraWasher 8262<br/>Description: Superior performance<br/>and durability<br/>Price: $121.24<br/>Warranty: 6 months"]
    end

    subgraph "Text Preparation"
        TXT["Formatted string:<br/>'UltraWasher 8262: Offers superior<br/>performance and durability.<br/>Price: $121.24.<br/>Warranty: 6 months.'"]
    end

    subgraph "Voyage AI API"
        EMB["voyage-3 model<br/>Returns 1024-dim vector<br/>[0.123, -0.456, 0.789, ...]"]
    end

    subgraph "PostgreSQL + pgvector"
        DB["product_embeddings table<br/>product_id: UUID (FK)<br/>embedding: vector(1024)<br/>IVFFlat index"]
    end

    PR --> TXT --> EMB --> DB

    style PR fill:#336791,stroke:#264E6D,color:#fff
    style TXT fill:#6C757D,stroke:#495057,color:#fff
    style EMB fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style DB fill:#336791,stroke:#264E6D,color:#fff
```

Products are processed in batches of 10 with a 21-second pause between batches (Voyage AI free-tier limit: 3 requests per minute). The script tracks which products already have embeddings to support resume-on-failure.

### Query-Time Search — The Full Chain

When a user asks _"Tell me about UltraWasher"_, the ReACT engine calls `product_search`. Here is the exact path through all four services:

```mermaid
sequenceDiagram
    participant Agent as Agent Service
    participant Tool as Tool Service
    participant Know as Knowledge Service
    participant Voyage as Voyage AI API
    participant PG as PostgreSQL + pgvector

    Note over Agent: ReACT Iteration 1
    Agent->>Agent: LLM outputs Action: product_search({"query":"UltraWasher"})
    Agent->>Tool: gRPC ExecuteTool(tool_name="product_search", params)

    Note over Tool: tool_product_search handler
    Tool->>Know: gRPC RetrieveRelevantDocs(query="UltraWasher", top_k=5)

    Note over Know: Step 1: Embed the query
    Know->>Voyage: vo_client.embed(["UltraWasher"], model="voyage-3")
    Voyage-->>Know: [[0.118, -0.442, 0.801, ...]]  (1024 floats)

    Note over Know: Step 2: pgvector cosine search
    Know->>PG: SELECT p.*, 1-(pe.embedding <=> query_vec) AS similarity FROM product_embeddings pe JOIN products p ON pe.product_id = p.id ORDER BY pe.embedding <=> query_vec LIMIT 5
    PG-->>Know: UltraWasher 8262 (0.952), MegaBlender 4455 (0.341), ...

    Note over Know: Step 3: Build response
    Know-->>Tool: KnowledgeResponse(products=[ProductDocument, ...])

    Note over Tool: Format as JSON
    Tool-->>Agent: {"results":[{"product_name":"UltraWasher 8262","similarity_score":0.952,...}],"count":1}

    Note over Agent: Observation fed back into ReACT loop
    Agent->>Agent: LLM sees result, produces Answer on next iteration
```

### The Cosine Similarity Math

The pgvector `<=>` operator computes cosine distance between the query vector and each stored product vector:

```
cosine_distance(A, B) = 1 - (A . B) / (||A|| x ||B||)
```

The SQL then converts distance to a similarity score:

```sql
1 - (pe.embedding <=> query_vector) AS similarity
```

```mermaid
flowchart LR
    subgraph "Query Vector"
        QV["embed('UltraWasher')<br/>[0.118, -0.442, 0.801, ...]"]
    end

    subgraph "Product Vectors in DB"
        PV1["UltraWasher 8262<br/>[0.121, -0.438, 0.799, ...]"]
        PV2["RoboCleaner 3000<br/>[-0.305, 0.612, -0.114, ...]"]
        PV3["EcoKettle 7200<br/>[0.445, -0.021, 0.337, ...]"]
    end

    subgraph "Cosine Distance"
        D1["distance = 0.048<br/>similarity = 0.952"]
        D2["distance = 0.816<br/>similarity = 0.184"]
        D3["distance = 0.659<br/>similarity = 0.341"]
    end

    subgraph "Ranked Results"
        R["1. UltraWasher 8262 (0.952)<br/>2. EcoKettle 7200 (0.341)<br/>3. RoboCleaner 3000 (0.184)"]
    end

    QV --> D1
    PV1 --> D1
    QV --> D2
    PV2 --> D2
    QV --> D3
    PV3 --> D3
    D1 --> R
    D2 --> R
    D3 --> R

    style QV fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style PV1 fill:#28A745,stroke:#1E7E34,color:#fff
    style PV2 fill:#6C757D,stroke:#495057,color:#fff
    style PV3 fill:#6C757D,stroke:#495057,color:#fff
    style D1 fill:#28A745,stroke:#1E7E34,color:#fff
    style D2 fill:#DC3545,stroke:#A71D2A,color:#fff
    style D3 fill:#FFC107,stroke:#CC9A06,color:#000
    style R fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

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

```mermaid
flowchart TD
    PLAN["Planning Layer Output:<br/>needs_multi_agent=true<br/>agents: warranty_specialist,<br/>comparison_specialist"]

    subgraph "Specialist 1: warranty_specialist"
        A1_SYS["System Prompt:<br/>Base + Focus on warranty<br/>Preferred: warranty_check"]
        A1_LOOP["ReACT Sub-Loop<br/>(max 4 iterations)"]
        A1_OUT["Output: Warranty details<br/>for both products"]
    end

    subgraph "Specialist 2: comparison_specialist"
        A2_SYS["System Prompt:<br/>Base + Compare systematically<br/>Preferred: product_compare"]
        A2_LOOP["ReACT Sub-Loop<br/>(max 4 iterations)"]
        A2_OUT["Output: Side-by-side<br/>comparison analysis"]
    end

    SYNTH["LLM Synthesis Call<br/>Combine specialist outputs<br/>into unified response"]
    FRAME["Response Framing<br/>Polish + confidence + sources"]

    PLAN --> A1_SYS --> A1_LOOP --> A1_OUT
    PLAN --> A2_SYS --> A2_LOOP --> A2_OUT
    A1_OUT --> SYNTH
    A2_OUT --> SYNTH
    SYNTH --> FRAME

    style PLAN fill:#17A2B8,stroke:#117A8B,color:#fff
    style A1_SYS fill:#E83E8C,stroke:#B5305F,color:#fff
    style A1_LOOP fill:#E83E8C,stroke:#B5305F,color:#fff
    style A1_OUT fill:#E83E8C,stroke:#B5305F,color:#fff
    style A2_SYS fill:#6F42C1,stroke:#59359A,color:#fff
    style A2_LOOP fill:#6F42C1,stroke:#59359A,color:#fff
    style A2_OUT fill:#6F42C1,stroke:#59359A,color:#fff
    style SYNTH fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style FRAME fill:#20C997,stroke:#199B76,color:#fff
```

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

```mermaid
flowchart TD
    subgraph "REFLECTION (Stage 8)"
        direction TB
        R_IN["Framed Response"]
        R_EVAL["LLM Evaluate<br/>5 criteria, each 0.0 to 1.0:<br/>completeness, accuracy,<br/>relevance, clarity, actionability"]
        R_CHECK{"overall_score<br/>at or above 0.75?"}
        R_PASS["Pass through"]
        R_REFINE["LLM Refine<br/>Fix identified issues"]
        R_MAX{"Max iterations<br/>reached? (2)"}
        R_DONE["Response finalized"]
        R_SCOPE["Scope: This request only<br/>Nothing persists"]

        R_IN --> R_EVAL --> R_CHECK
        R_CHECK -->|"Yes"| R_PASS --> R_DONE
        R_CHECK -->|"No"| R_REFINE --> R_MAX
        R_MAX -->|"No"| R_EVAL
        R_MAX -->|"Yes"| R_DONE
    end

    subgraph "REFLEXION (Stage 10)"
        direction TB
        X_IN["Original Score<br/>from Reflection"]
        X_CHECK{"Score below<br/>0.7 threshold?"}
        X_SKIP["Skip, no storage needed"]
        X_GEN["LLM Self-Reflect<br/>Generate reusable insight:<br/>query_pattern, failure_reason,<br/>suggested_improvement, key_topics"]
        X_STORE[("Store to TimescaleDB<br/>event_type: reflexion_insight<br/>Persists permanently")]
        X_FUTURE["FUTURE QUERIES:<br/>Fetch matching insights<br/>Inject into ReACT prompt<br/>as 'Learnings from past'"]

        X_IN --> X_CHECK
        X_CHECK -->|"No"| X_SKIP
        X_CHECK -->|"Yes"| X_GEN --> X_STORE
        X_STORE -.->|"Retrieved on<br/>similar future queries"| X_FUTURE
    end

    style R_IN fill:#6610F2,stroke:#510EC0,color:#fff
    style R_EVAL fill:#6610F2,stroke:#510EC0,color:#fff
    style R_CHECK fill:#6610F2,stroke:#510EC0,color:#fff
    style R_PASS fill:#28A745,stroke:#1E7E34,color:#fff
    style R_REFINE fill:#6610F2,stroke:#510EC0,color:#fff
    style R_MAX fill:#6610F2,stroke:#510EC0,color:#fff
    style R_DONE fill:#28A745,stroke:#1E7E34,color:#fff
    style R_SCOPE fill:#6C757D,stroke:#495057,color:#fff

    style X_IN fill:#E65100,stroke:#BF4400,color:#fff
    style X_CHECK fill:#E65100,stroke:#BF4400,color:#fff
    style X_SKIP fill:#6C757D,stroke:#495057,color:#fff
    style X_GEN fill:#E65100,stroke:#BF4400,color:#fff
    style X_STORE fill:#E65100,stroke:#BF4400,color:#fff
    style X_FUTURE fill:#E65100,stroke:#BF4400,color:#fff
```

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

Reflection is the last chance to fix a response before the user sees it. It sits between Response Framing (Stage 7) and Output Guardrails (Stage 9). The system asks a separate LLM call _"Is this response actually good?"_ — and if not, a second LLM call improves it.

#### The Two LLM Roles

Reflection uses two distinct LLM calls, each with a different persona:

```mermaid
flowchart TD
    subgraph "LLM Call 1: The Critic"
        direction TB
        C_IN["Inputs:<br/>- Original user query<br/>- Framed response text<br/>- Tools used<br/>- Number of reasoning steps<br/>- Conversation context (500 chars)"]
        C_EVAL["Evaluator LLM<br/>temp=0.2, max_tokens=512<br/>System: 'You are a quality evaluator'"]
        C_OUT["Output: 5 scores (0.0-1.0)<br/>+ overall_score<br/>+ issues list<br/>+ suggestions list<br/>+ needs_refinement flag"]
    end

    subgraph "LLM Call 2: The Improver"
        direction TB
        I_IN["Inputs:<br/>- Original user query<br/>- Current response text<br/>- Critique (score + issues)<br/>- Raw tool observations<br/>  from ReACT loop"]
        I_REFINE["Refiner LLM<br/>temp=0.2, max_tokens=1024<br/>System: 'You are a response refiner'"]
        I_OUT["Output:<br/>- Improved response text<br/>- Updated confidence<br/>- Updated sources"]
    end

    C_IN --> C_EVAL --> C_OUT
    I_IN --> I_REFINE --> I_OUT

    style C_IN fill:#6610F2,stroke:#510EC0,color:#fff
    style C_EVAL fill:#6610F2,stroke:#510EC0,color:#fff
    style C_OUT fill:#6610F2,stroke:#510EC0,color:#fff
    style I_IN fill:#17A2B8,stroke:#117A8B,color:#fff
    style I_REFINE fill:#17A2B8,stroke:#117A8B,color:#fff
    style I_OUT fill:#17A2B8,stroke:#117A8B,color:#fff
```

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

```mermaid
flowchart TD
    START(["Enter Reflection Loop"])

    OBS["Compile tool observations<br/>from all ReACT steps"]

    ITER{"Iteration i<br/>(max 2)"}

    EVAL["LLM Evaluate<br/>Score response on 5 criteria"]
    EMIT_EVAL["Emit: reflection_evaluating"]
    EMIT_CRIT["Emit: reflection_critique<br/>(score, issues, suggestions)"]

    CHECK_SCORE{"overall_score<br/>at or above 0.75?"}
    EXIT_GOOD(["EXIT: Quality sufficient<br/>Use current response"])

    CHECK_FLAG{"needs_refinement<br/>== true?"}
    EXIT_FLAG(["EXIT: Evaluator says<br/>no refinement needed<br/>Use current response"])

    REFINE["LLM Refine<br/>Fix identified issues using<br/>critique + tool observations"]
    EMIT_REF["Emit: reflection_refining"]

    CHECK_PARSE{"Refine returned<br/>valid JSON?"}
    EXIT_PARSE(["EXIT: Parse failure<br/>Keep previous response"])

    UPDATE["Replace current response<br/>with refined version"]

    START --> OBS --> ITER
    ITER --> EMIT_EVAL --> EVAL --> EMIT_CRIT --> CHECK_SCORE
    CHECK_SCORE -->|"Yes"| EXIT_GOOD
    CHECK_SCORE -->|"No"| CHECK_FLAG
    CHECK_FLAG -->|"No"| EXIT_FLAG
    CHECK_FLAG -->|"Yes"| EMIT_REF --> REFINE --> CHECK_PARSE
    CHECK_PARSE -->|"No"| EXIT_PARSE
    CHECK_PARSE -->|"Yes"| UPDATE --> ITER

    style START fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style OBS fill:#6C757D,stroke:#495057,color:#fff
    style EVAL fill:#6610F2,stroke:#510EC0,color:#fff
    style REFINE fill:#17A2B8,stroke:#117A8B,color:#fff
    style CHECK_SCORE fill:#FFC107,stroke:#CC9A06,color:#000
    style CHECK_FLAG fill:#FFC107,stroke:#CC9A06,color:#000
    style CHECK_PARSE fill:#FFC107,stroke:#CC9A06,color:#000
    style EXIT_GOOD fill:#28A745,stroke:#1E7E34,color:#fff
    style EXIT_FLAG fill:#28A745,stroke:#1E7E34,color:#fff
    style EXIT_PARSE fill:#6C757D,stroke:#495057,color:#fff
    style UPDATE fill:#17A2B8,stroke:#117A8B,color:#fff
    style EMIT_EVAL fill:#6610F2,stroke:#510EC0,color:#fff
    style EMIT_CRIT fill:#6610F2,stroke:#510EC0,color:#fff
    style EMIT_REF fill:#17A2B8,stroke:#117A8B,color:#fff
```

#### Concrete Example: A Poor Response Gets Fixed

Query: _"Compare the warranty of UltraWasher 8262 with RoboCleaner 3000"_

The ReACT loop only checked one product and produced a partial answer. After framing, the response is: _"The UltraWasher 8262 has a 6-month warranty."_ (missing the RoboCleaner comparison entirely).

```mermaid
sequenceDiagram
    participant RefL as Reflection Loop
    participant Critic as Evaluator LLM
    participant Improver as Refiner LLM

    Note over RefL: Iteration 1

    RefL->>Critic: Evaluate: query="Compare warranty..."<br/>response="The UltraWasher 8262 has a 6-month warranty."<br/>tools=["warranty_check"]
    Critic-->>RefL: completeness=0.3, accuracy=0.8, relevance=0.5,<br/>clarity=0.7, actionability=0.2<br/>overall_score=0.50<br/>issues=["Only one product discussed",<br/>"No RoboCleaner warranty info"]<br/>needs_refinement=true

    Note over RefL: Score 0.50 < 0.75 AND needs_refinement=true

    RefL->>Improver: Refine with critique + tool observations:<br/>"warranty_check: {warranty_months:6, price:121.24}"<br/>"warranty_check: {warranty_months:36, price:249.99}"
    Improver-->>RefL: "The UltraWasher 8262 has a 6-month warranty<br/>(from Sept 2023, valid through March 2024),<br/>while the RoboCleaner 3000 offers a longer<br/>36-month warranty (from Feb 2024, valid through<br/>Feb 2027). The RoboCleaner provides 30 months<br/>more coverage."<br/>confidence=0.89

    Note over RefL: Response replaced with refined version

    Note over RefL: Iteration 2

    RefL->>Critic: Evaluate the refined response
    Critic-->>RefL: completeness=0.9, accuracy=0.9, relevance=0.95,<br/>clarity=0.9, actionability=0.8<br/>overall_score=0.89<br/>needs_refinement=false

    Note over RefL: Score 0.89 >= 0.75 -> EXIT<br/>Refined response sent to user
```

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

#### Connection to Reflexion (Stage 10)

Reflection produces two values that Reflexion consumes downstream:

| Value             | Source                                          | Used By Reflexion                                    |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------- |
| `original_score`  | First evaluation score (before any refinement)  | Gate: if `< 0.7`, store a persistent insight         |
| `last_evaluation` | The evaluation dict with issues and suggestions | Passed to LLM self-reflection for insight generation |

Reflection is the **scoring mechanism** that determines whether Reflexion fires. Without Reflection, Reflexion would have no quality signal to act on.

### Reflexion Lifecycle — Write and Read Paths

```mermaid
sequenceDiagram
    participant User as User Query
    participant React as ReACT Loop
    participant Reflect as Reflection
    participant Reflexion as Reflexion Engine
    participant TSDB as TimescaleDB
    participant Future as Future Query

    Note over User,Future: WRITE PATH (current request with poor quality)

    User->>React: "What warranty does ProductX have?"
    React->>React: Tool call returns error (product not found)
    React->>React: Gives incomplete answer
    React->>Reflect: Framed response
    Reflect->>Reflect: Evaluate: score = 0.55
    Reflect->>Reflect: Refine: improved score = 0.72
    Reflect->>Reflexion: original_score=0.55 (below 0.7)
    Reflexion->>Reflexion: LLM Self-Reflect: generate insight
    Note right of Reflexion: {"query_pattern": "warranty lookup",<br/>"failure_reason": "Product name was<br/>not found, needed fuzzy match",<br/>"suggested_improvement": "Try product<br/>search first to get exact name",<br/>"key_topics": ["warranty", "ProductX"]}
    Reflexion->>TSDB: Store as reflexion_insight

    Note over User,Future: READ PATH (future similar query)

    Future->>React: "Check warranty for ProductY"
    React->>TSDB: Fetch reflexion_insights (intent=warranty_question)
    TSDB-->>React: Past insight: "Try product search first"
    Note right of React: System prompt now includes:<br/>"Learnings from past interactions:<br/>- Try product search first to get<br/>  exact name before warranty lookup"
    React->>React: Iteration 1: product_search("ProductY")
    React->>React: Iteration 2: warranty_check("ProductY 5500")
    React->>React: Complete, accurate answer
    React->>Reflect: Evaluate: score = 0.92
    Reflect->>Reflexion: original_score=0.92 (above 0.7)
    Reflexion->>Reflexion: Skip storage (quality was good)
```

### Reflexion Deep Dive — Persistent Cross-Session Learning

Reflexion implements the academic Reflexion pattern (Shinn et al., 2023): an agent that reflects on failures, generates verbal reinforcement, stores it in persistent memory, and retrieves it on future similar queries — improving performance without retraining the model.

#### The Two Independent Code Paths

The write path and read path are completely decoupled. They run at different times and don't depend on each other within a single request.

```mermaid
flowchart TD
    subgraph "WRITE PATH (Stage 10 — end of request)"
        direction TB
        W_GATE{"original_score<br/>below 0.7?"}
        W_SKIP["Skip storage<br/>(quality was good)"]
        W_LLM["LLM Self-Reflect<br/>System: 'You are a<br/>self-reflection agent'<br/>temp=0.2, max_tokens=512"]
        W_PARSE{"Valid JSON<br/>returned?"}
        W_FAIL["Skip storage<br/>(parse failure)"]
        W_STORE[("StoreEpisodicMemory<br/>event_type: reflexion_insight<br/>summary: the improvement text<br/>key_topics: for future matching<br/>metadata: scores + failure reason")]
        W_EVENT["Emit: reflexion_learning"]

        W_GATE -->|"No"| W_SKIP
        W_GATE -->|"Yes"| W_LLM --> W_PARSE
        W_PARSE -->|"No"| W_FAIL
        W_PARSE -->|"Yes"| W_STORE --> W_EVENT
    end

    subgraph "READ PATH (before ReACT loop — start of request)"
        direction TB
        R_FETCH["GetEpisodicMemories<br/>customer_id, limit=5<br/>event_type=reflexion_insight"]
        R_EMPTY{"Memories<br/>found?"}
        R_NONE["Return empty string<br/>(no insights to inject)"]
        R_FILTER["Filter by relevance:<br/>1. Intent match<br/>2. Topic word overlap<br/>3. Fallback: most recent"]
        R_FORMAT["Format top 3 as:<br/>'Learnings from past interactions:<br/>- insight summary 1<br/>- insight summary 2<br/>- insight summary 3'"]
        R_INJECT["Inject into ReACT<br/>system prompt via<br/>{reflexion_context} placeholder"]

        R_FETCH --> R_EMPTY
        R_EMPTY -->|"No"| R_NONE
        R_EMPTY -->|"Yes"| R_FILTER --> R_FORMAT --> R_INJECT
    end

    style W_GATE fill:#E65100,stroke:#BF4400,color:#fff
    style W_SKIP fill:#6C757D,stroke:#495057,color:#fff
    style W_LLM fill:#E65100,stroke:#BF4400,color:#fff
    style W_PARSE fill:#E65100,stroke:#BF4400,color:#fff
    style W_FAIL fill:#6C757D,stroke:#495057,color:#fff
    style W_STORE fill:#E65100,stroke:#BF4400,color:#fff
    style W_EVENT fill:#E65100,stroke:#BF4400,color:#fff
    style R_FETCH fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style R_EMPTY fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style R_NONE fill:#6C757D,stroke:#495057,color:#fff
    style R_FILTER fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style R_FORMAT fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style R_INJECT fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

#### Write Path — Generating and Storing Insights

The write path runs at Stage 10, **after** the response has already been streamed to the user. It is entirely non-blocking — failures are logged and swallowed.

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

```mermaid
flowchart LR
    subgraph "Insight Record in TimescaleDB"
        F1["event_type:<br/>reflexion_insight"]
        F2["summary:<br/>'Always use product_search first<br/>to get exact product name, then<br/>call warranty_check with the<br/>matched name'"]
        F3["key_topics:<br/>['warranty', 'product_search',<br/>'ProductX']"]
        F4["resolution_status:<br/>'resolved' if refinement helped<br/>'unresolved' if it did not"]
        F5["metadata (JSON):<br/>query_pattern, intent,<br/>failure_reason, original_score,<br/>refined_score, tools_used"]
    end

    style F1 fill:#E65100,stroke:#BF4400,color:#fff
    style F2 fill:#28A745,stroke:#1E7E34,color:#fff
    style F3 fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style F4 fill:#6C757D,stroke:#495057,color:#fff
    style F5 fill:#6C757D,stroke:#495057,color:#fff
```

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

```mermaid
flowchart TD
    INSIGHTS["5 past insights fetched<br/>from TimescaleDB"]

    M1{"Intent match?<br/>stored intent == current intent"}
    M2{"Topic overlap?<br/>query words intersect key_topics"}
    M3["Fallback:<br/>use most recent insights"]
    RESULT["Top 3 relevant insights<br/>formatted for prompt injection"]

    INSIGHTS --> M1
    M1 -->|"Matched"| RESULT
    M1 -->|"No match"| M2
    M2 -->|"Matched"| RESULT
    M2 -->|"No match"| M3 --> RESULT

    style INSIGHTS fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style M1 fill:#28A745,stroke:#1E7E34,color:#fff
    style M2 fill:#FFC107,stroke:#CC9A06,color:#000
    style M3 fill:#6C757D,stroke:#495057,color:#fff
    style RESULT fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

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

```mermaid
sequenceDiagram
    participant User1 as Session 1 (Tuesday)
    participant React1 as ReACT Loop
    participant Reflect1 as Reflection
    participant Reflexion1 as Reflexion Engine
    participant TSDB as TimescaleDB
    participant User2 as Session 2 (Wednesday)
    participant React2 as ReACT Loop

    Note over User1,React1: Session 1: Agent fails on warranty lookup

    User1->>React1: "What warranty does ProductX have?"
    React1->>React1: Thought: Check warranty directly
    React1->>React1: Action: warranty_check({"product_name":"ProductX"})
    React1->>React1: Observation: {"results":[],"count":0}
    Note right of React1: ProductX not found!<br/>Exact name is "ProductX 5500"
    React1->>React1: Answer: "I couldn't find warranty info for ProductX"

    React1->>Reflect1: Framed response
    Reflect1->>Reflect1: Evaluate: score = 0.45
    Reflect1->>Reflect1: Refine: improved text, score = 0.72

    Note over Reflect1,Reflexion1: original_score 0.45 < 0.7 threshold

    Reflect1->>Reflexion1: original_score=0.45, issues, evaluation
    Reflexion1->>Reflexion1: LLM Self-Reflect: analyze failure
    Note right of Reflexion1: Insight generated:<br/>"Always use product_search first<br/>to resolve the exact product name"
    Reflexion1->>TSDB: Store reflexion_insight
    Note right of TSDB: summary: "Always use product_search<br/>first to resolve exact name"<br/>key_topics: ["warranty","ProductX"]<br/>intent: "warranty_question"

    Note over User2,React2: Session 2 (next day): Similar query

    User2->>React2: "Check warranty for ProductY"
    React2->>TSDB: GetEpisodicMemories(event_type=reflexion_insight)
    TSDB-->>React2: 1 insight matched (intent=warranty_question)

    Note right of React2: System prompt now includes:<br/>"Learnings from past interactions:<br/>- Always use product_search first<br/>  to resolve exact product name"

    React2->>React2: Thought: Based on past learnings, I should<br/>search for the exact name first
    React2->>React2: Action: product_search({"query":"ProductY"})
    React2->>React2: Observation: {"results":[{"product_name":"ProductY 7200"}]}
    React2->>React2: Thought: Found exact name. Now check warranty.
    React2->>React2: Action: warranty_check({"product_name":"ProductY 7200"})
    React2->>React2: Observation: {"results":[{"warranty_months":24}]}
    React2->>React2: Answer: "ProductY 7200 has a 24-month warranty..."

    React2->>Reflect1: Evaluate: score = 0.91
    Note over Reflect1,Reflexion1: 0.91 >= 0.7 -> skip storage<br/>Nothing to learn this time
```

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

The recommendation service generates contextual suggestions at two points: session start and after each response.

### 3-Tier Cold Start Strategy

```mermaid
flowchart TD
    START(["GetStartRecommendations"])

    T1{"Tier 1:<br/>Returning user?<br/>Has episodic memories?"}
    T1_YES["Build Customer Profile<br/>Gap analysis vs empty session<br/>Personalized suggestions:<br/>1. Continue from last topic<br/>2. Unexplored brand + price range<br/>3. Missing intent with real data"]

    T2{"Tier 2:<br/>System has data?<br/>Other users' conversations?"}
    T2_YES["Cross-User Popular Queries<br/>Aggregated from conversation_turns<br/>Top 5 by frequency"]

    T3["Tier 3: Catalog-Aware Defaults<br/>Real brand names from products table<br/>Real price ranges from catalog"]

    OUT(["Return up to 5 suggestions"])

    START --> T1
    T1 -->|"Yes"| T1_YES --> OUT
    T1 -->|"No"| T2
    T2 -->|"Yes"| T2_YES --> OUT
    T2 -->|"No"| T3 --> OUT

    style START fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style T1 fill:#28A745,stroke:#1E7E34,color:#fff
    style T1_YES fill:#28A745,stroke:#1E7E34,color:#fff
    style T2 fill:#FFC107,stroke:#CC9A06,color:#000
    style T2_YES fill:#FFC107,stroke:#CC9A06,color:#000
    style T3 fill:#6C757D,stroke:#495057,color:#fff
    style OUT fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

### Follow-Up Recommendation Pipeline

After every response, the system generates exactly 3 product-focused follow-up suggestions:

```mermaid
flowchart LR
    A["Build Session Context<br/>intents, products, brands,<br/>tools from conversation_turns"] --> B["Build Customer Profile<br/>from episodic memories"]
    B --> C["Gap Analysis<br/>what user HAS done<br/>vs what they COULD do"]
    C --> D["Select Template Set<br/>keyed by (intent, gap_type)"]
    D --> E["Fill Templates<br/>with real product data"]
    E --> F["Deduplicate<br/>+ exclude last query echo"]
    F --> G{"3 suggestions?"}
    G -->|"No"| H["Gap-based fallbacks<br/>then catalog generics"]
    H --> G
    G -->|"Yes"| I(["Return 3 suggestions"])

    style A fill:#50C878,stroke:#3A9A5C,color:#fff
    style B fill:#FF6B6B,stroke:#CC5555,color:#fff
    style C fill:#E65100,stroke:#BF4400,color:#fff
    style D fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style E fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style F fill:#6C757D,stroke:#495057,color:#fff
    style G fill:#FFC107,stroke:#CC9A06,color:#000
    style H fill:#6C757D,stroke:#495057,color:#fff
    style I fill:#4A90D9,stroke:#2C5F8A,color:#fff
```

#### Gap Analysis Dimensions

| Dimension   | Full Set                                                      | Gap = Set Difference                                              |
| ----------- | ------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Intents** | product_inquiry, price_check, warranty_question, comparison   | Unused intents, priority: comparison > warranty > price > inquiry |
| **Brands**  | All brands from product catalog                               | Brands not yet mentioned, sorted by product count                 |
| **Tools**   | product_search, price_lookup, warranty_check, product_compare | Tools not yet triggered                                           |

---

## 10. Data Storage Architecture

### Memory Layers

Memory is organized into five layers across three storage engines. The Memory Service provides a unified gRPC interface, routing reads and writes to the appropriate backend.

```mermaid
graph LR
    subgraph "Hot Path (Redis)"
        ST["Short-Term Memory<br/>Current session turns<br/>TTL: 30min sliding"]
    end

    subgraph "Immutable Path (TimescaleDB)"
        EP["Episodic Memory<br/>Session summaries<br/>Reflexion insights<br/>Evaluation records<br/>Hypertable, append-only"]
        AT["Audit Trail<br/>Every session event<br/>Hypertable, append-only"]
    end

    subgraph "Relational Path (PostgreSQL + pgvector)"
        SM["Semantic Memory<br/>Product embeddings<br/>pgvector index"]
        PR["Procedural Memory<br/>Tool definitions<br/>and usage patterns"]
        DM["Domain Memory<br/>Product catalog<br/>and metadata"]
    end

    ST -->|"session ends"| EP
    ST -->|"every event"| AT

    style ST fill:#DC382D,stroke:#A82B23,color:#fff
    style EP fill:#E65100,stroke:#BF4400,color:#fff
    style AT fill:#E65100,stroke:#BF4400,color:#fff
    style SM fill:#336791,stroke:#264E6D,color:#fff
    style PR fill:#336791,stroke:#264E6D,color:#fff
    style DM fill:#336791,stroke:#264E6D,color:#fff
```

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Short-Term** | Redis hash per session, TTL 30min sliding | Current conversation turns, ReACT state |
| **Episodic** | TimescaleDB hypertable (7-day chunks, immutable) | Session summaries, reflexion insights, evaluation records |
| **Audit Trail** | TimescaleDB hypertable (1-day chunks, 90-day retention) | Complete event log for session replay and compliance |
| **Semantic** | PostgreSQL + pgvector (IVFFlat index) | Product embeddings for similarity search |
| **Procedural / Domain** | PostgreSQL | Tool definitions, product catalog, user data |

### Storage Schema

```mermaid
flowchart TD
    subgraph "PostgreSQL (Relational, Durable)"
        PG_SESSIONS["sessions<br/>id, customer_id (FK),<br/>metadata (JSONB),<br/>created_at, last_active_at"]
        PG_TURNS["conversation_turns<br/>id, session_id (FK), role,<br/>content, intent, confidence,<br/>tool_calls (JSONB), created_at"]
        PG_PRODUCTS["products<br/>id, product_name, description,<br/>price, manufacturing_date,<br/>warranty_months, created_at"]
        PG_TOOLS["tool_definitions<br/>id, name, description,<br/>parameter_schema (JSON),<br/>is_active"]
        PG_CUSTOMERS["customers<br/>id, name, email,<br/>password_hash, created_at"]
    end

    subgraph "TimescaleDB (Time-Series, Immutable)"
        TS_EPISODIC["episodic_memories<br/>id, customer_id, session_id,<br/>event_type, summary,<br/>key_topics (TEXT[]),<br/>resolution_status,<br/>metadata (JSONB), created_at"]
        TS_AUDIT["session_audit_trail<br/>id, session_id, customer_id,<br/>event_type, event_data (JSONB),<br/>event_time"]
    end

    subgraph "Redis (Cache, Volatile)"
        RD_SESSION["Session Cache<br/>TTL: 1800s<br/>Key: session:{id}"]
        RD_HISTORY["Conversation Cache<br/>List per session<br/>Key: history:{session_id}"]
    end

    style PG_SESSIONS fill:#336791,stroke:#264E6D,color:#fff
    style PG_TURNS fill:#336791,stroke:#264E6D,color:#fff
    style PG_PRODUCTS fill:#336791,stroke:#264E6D,color:#fff
    style PG_TOOLS fill:#336791,stroke:#264E6D,color:#fff
    style PG_CUSTOMERS fill:#336791,stroke:#264E6D,color:#fff
    style TS_EPISODIC fill:#E65100,stroke:#BF4400,color:#fff
    style TS_AUDIT fill:#E65100,stroke:#BF4400,color:#fff
    style RD_SESSION fill:#DC382D,stroke:#A82B23,color:#fff
    style RD_HISTORY fill:#DC382D,stroke:#A82B23,color:#fff
```

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

```mermaid
erDiagram
    USERS {
        uuid id PK
        varchar email UK
        varchar password_hash
        varchar display_name
        boolean is_active
        timestamp created_at
        timestamp last_login_at
    }

    PRODUCTS {
        uuid id PK
        varchar product_name
        text description
        decimal price
        date manufacturing_date
        int warranty_months
        timestamp created_at
    }

    PRODUCT_EMBEDDINGS {
        uuid id PK
        uuid product_id FK
        vector embedding
        timestamp created_at
    }

    SESSIONS {
        uuid id PK
        uuid customer_id FK
        jsonb metadata
        timestamp created_at
        timestamp last_active_at
    }

    CONVERSATION_TURNS {
        uuid id PK
        uuid session_id FK
        varchar role
        text content
        varchar intent
        float confidence
        jsonb tool_calls
        timestamp created_at
    }

    QUERY_PATTERNS {
        uuid id PK
        text query_text
        varchar intent
        int frequency
        timestamp last_seen
    }

    TOOL_DEFINITIONS {
        uuid id PK
        varchar name UK
        text description
        jsonb parameter_schema
        boolean is_active
        timestamp created_at
    }

    TOOL_EXECUTION_LOGS {
        uuid id PK
        uuid session_id FK
        uuid tool_id FK
        jsonb input_params
        jsonb output_result
        int execution_time_ms
        varchar status
        timestamp created_at
    }

    USERS ||--o{ SESSIONS : has
    PRODUCTS ||--o{ PRODUCT_EMBEDDINGS : has
    SESSIONS ||--o{ CONVERSATION_TURNS : contains
    SESSIONS ||--o{ TOOL_EXECUTION_LOGS : logs
    TOOL_DEFINITIONS ||--o{ TOOL_EXECUTION_LOGS : executed_by
```

#### TimescaleDB (Immutable Hypertables)

```mermaid
erDiagram
    EPISODIC_MEMORIES {
        uuid id PK
        varchar customer_id
        uuid session_id
        varchar event_type
        text summary
        text_arr key_topics
        varchar resolution_status
        jsonb metadata
        timestamptz created_at PK
    }

    SESSION_AUDIT_TRAIL {
        uuid id PK
        uuid session_id
        varchar customer_id
        varchar event_type
        jsonb event_data
        timestamptz event_time PK
    }

    DAILY_SESSION_STATS {
        timestamptz day
        varchar customer_id
        int sessions_started
        int user_messages
        int tool_executions
        int clarifications
        int errors
    }

    SESSION_AUDIT_TRAIL }|--|| DAILY_SESSION_STATS : aggregates
```

Both TimescaleDB tables are **append-only** — UPDATE and DELETE are blocked by database triggers. `episodic_memories` uses 7-day chunks; `session_audit_trail` uses 1-day chunks with a 90-day retention policy. `daily_session_stats` is a continuous aggregate materialised view, auto-refreshed hourly.

---

## 11. End-to-End Query Walkthrough

### Example: "Compare the warranty of UltraWasher 8262 with RoboCleaner 3000"

This walkthrough traces every stage, event, and service call for a real query.

```mermaid
sequenceDiagram
    actor User
    participant GW as Gateway
    participant AS as Agent Service
    participant LLM as LLM Service
    participant MS as Memory Service
    participant TS as Tool Service
    participant RS as Recommendation Svc
    participant DB as PostgreSQL
    participant TSDB as TimescaleDB

    User->>GW: WebSocket: {"type":"user_message", "query":"Compare the warranty..."}
    GW->>GW: JWT validation + rate limit check
    GW->>AS: gRPC ProcessQuery(session_id, customer_id, query)

    Note over AS: Stage 0: Session Context
    AS->>MS: TouchSession(session_id)
    MS->>DB: UPDATE sessions SET last_active_at
    AS->>MS: GetConversationHistory(session_id, limit=10)
    MS-->>AS: Last 10 turns
    AS->>MS: AddConversationTurn(role=user, content=query)

    Note over AS: Stage 1: Input Guardrails
    AS->>AS: Regex checks: length OK, no injection, no PII

    Note over AS: Stage 2: Intent Classification
    AS->>LLM: GenerateAnswer(INTENT_PROMPT, temp=0.1)
    LLM-->>AS: {"intent":"comparison","confidence":0.94}

    Note over AS: Stage 3: Clarification
    AS->>AS: confidence 0.94 >= 0.8, skip clarification

    Note over AS: Stage 4: Planning
    AS->>TS: ListTools()
    TS-->>AS: [product_search, price_lookup, warranty_check, product_compare]
    AS->>LLM: GenerateAnswer(PLANNING_PROMPT, temp=0.2)
    LLM-->>AS: needs_multi_agent=false, steps=[warranty x2, compare]
    AS-->>GW: Event: agent_planning

    Note over AS: Stage 5: ReACT Loop (Single Agent)
    AS->>TSDB: GetEpisodicMemories(event_type=reflexion_insight)
    TSDB-->>AS: 1 past insight: "Check both products before answering"
    AS->>AS: Inject insight + plan into system prompt

    rect rgb(30, 60, 100)
        Note over AS,TS: ReACT Iteration 1
        AS->>LLM: ReACT step (temp=0.3)
        LLM-->>AS: Thought: Need UltraWasher warranty. Action: warranty_check({"product_name":"UltraWasher 8262"})
        AS->>AS: Validate params: OK
        AS->>TS: ExecuteTool(warranty_check, params)
        TS->>DB: SELECT warranty_months, manufacturing_date FROM products
        TS-->>AS: {"warranty_months":24,"manufacturing_date":"2024-01-15"}
        AS->>AS: Validate result: OK
        AS-->>GW: Event: agent_thinking (iteration 1)
    end

    rect rgb(30, 60, 100)
        Note over AS,TS: ReACT Iteration 2
        AS->>LLM: ReACT step with history
        LLM-->>AS: Thought: Now check RoboCleaner. Action: warranty_check({"product_name":"RoboCleaner 3000"})
        AS->>TS: ExecuteTool(warranty_check, params)
        TS-->>AS: {"warranty_months":36,"manufacturing_date":"2024-02-10"}
        AS-->>GW: Event: agent_thinking (iteration 2)
    end

    rect rgb(30, 60, 100)
        Note over AS,TS: ReACT Iteration 3
        AS->>LLM: ReACT step with history
        LLM-->>AS: Thought: I have both warranties. Answer: RoboCleaner 3000 has 36 months vs UltraWasher 8262's 24 months...
        AS-->>GW: Event: agent_thinking (iteration 3)
    end

    Note over AS: Stage 7: Response Framing
    AS->>LLM: GenerateAnswer(FRAMING_PROMPT, temp=0.2)
    LLM-->>AS: {"text":"polished response","confidence":0.91,"sources":["UltraWasher 8262","RoboCleaner 3000"]}

    Note over AS: Stage 8: Reflection
    AS->>LLM: GenerateAnswer(EVALUATE_PROMPT, temp=0.2)
    LLM-->>AS: {"overall_score":0.88,"needs_refinement":false}
    AS-->>GW: Event: reflection_evaluating
    AS-->>GW: Event: reflection_critique (score=0.88)

    Note over AS: Stage 9: Output Guardrails
    AS->>AS: PII scan: clean

    Note over AS: Stage 10: Reflexion
    AS->>AS: original_score 0.88 >= 0.7, skip insight storage

    Note over AS: Stage 11: Evaluation Storage
    AS->>TSDB: StoreEpisodicMemory(event_type=evaluation_record)

    Note over AS: Stage 12: Recommendations + Streaming
    AS->>RS: GetFollowUpRecommendations(session_id, intent, customer_id)
    RS-->>AS: ["What's the price of RoboCleaner 3000?","Explore EcoKettle products","Which product has the longest warranty?"]
    AS-->>GW: Event: token (streamed response text)
    AS-->>GW: Event: response_complete (response + recommendations)
    GW-->>User: WebSocket: streamed tokens + final payload
```

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

 Suggestions  "What's the price of RoboCleaner 3000?"
              "Explore EcoKettle products"
              "Which product has the longest warranty?"
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

```mermaid
graph TB
    subgraph Docker Compose
        subgraph Infrastructure
            REDIS["Redis 7<br/>:6379"]
            PG["PostgreSQL 16<br/>+ pgvector<br/>:5432"]
            TS["TimescaleDB<br/>:5433<br/>Immutable Store"]
        end

        subgraph Application Services
            GW["Gateway Service<br/>:8000<br/>FastAPI + WS"]
            AG["Agent Service<br/>:50054<br/>gRPC"]
            MEM["Memory Service<br/>:50055<br/>gRPC"]
            LLM_S["LLM Service<br/>:50053<br/>gRPC"]
            KN["Knowledge Service<br/>:50052<br/>gRPC"]
            TOOL_S["Tool Service<br/>:50056<br/>gRPC"]
            REC["Recommendation Service<br/>:50057<br/>gRPC"]
        end
    end

    GW --> AG
    AG --> MEM
    AG --> LLM_S
    AG --> KN
    AG --> TOOL_S
    AG --> REC
    MEM --> REDIS
    MEM --> PG
    MEM --> TS
    KN --> PG
    TOOL_S --> PG
    REC --> PG

    style GW fill:#F5A623,stroke:#C47D0E,color:#fff
    style AG fill:#D0021B,stroke:#9B0016,color:#fff
    style MEM fill:#FF6B6B,stroke:#CC5555,color:#fff
    style LLM_S fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style KN fill:#7B68EE,stroke:#5A4ACB,color:#fff
    style TOOL_S fill:#50C878,stroke:#3A9A5C,color:#fff
    style REC fill:#50C878,stroke:#3A9A5C,color:#fff
    style REDIS fill:#DC382D,stroke:#A82B23,color:#fff
    style PG fill:#336791,stroke:#264E6D,color:#fff
    style TS fill:#336791,stroke:#264E6D,color:#fff
```

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

```mermaid
sequenceDiagram
    participant B as Browser
    participant GW as Gateway
    participant PG as PostgreSQL

    B->>GW: POST /api/login {email, password}
    GW->>PG: SELECT user WHERE email = ?
    PG-->>GW: user row (id, password_hash, ...)
    GW->>GW: bcrypt.verify(password, hash)
    GW-->>B: {token: "eyJ...", user: {...}}

    B->>GW: WebSocket /ws/chat
    B->>GW: {type: "session_start", token: "eyJ..."}
    GW->>GW: JWT.verify(token)
    GW->>GW: Extract user_id from claims
    GW-->>B: {type: "session_ready", session_id: "..."}
```

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

```mermaid
graph TB
    subgraph "TLS Certificate Chain"
        CA["Self-Signed CA<br/>ca.pem / ca-key.pem"]
        SC["Server Certificate<br/>server.pem / server-key.pem<br/>SANs: all service hostnames"]
    end

    CA -->|"signs"| SC

    subgraph "gRPC Services (TLS Server)"
        AG["Agent :50054"]
        MEM_S["Memory :50055"]
        LLM_S2["LLM :50053"]
        KN_S["Knowledge :50052"]
        TOOL_S2["Tool :50056"]
        REC_S["Recommendation :50057"]
    end

    subgraph "gRPC Clients (TLS Channel)"
        GW_C["Gateway"]
        AG_C["Agent"]
        TOOL_C["Tool"]
        REC_C["Recommendation"]
    end

    SC -->|"loaded by"| AG
    SC -->|"loaded by"| MEM_S
    SC -->|"loaded by"| LLM_S2
    SC -->|"loaded by"| KN_S
    SC -->|"loaded by"| TOOL_S2
    SC -->|"loaded by"| REC_S

    CA -->|"trusted by"| GW_C
    CA -->|"trusted by"| AG_C
    CA -->|"trusted by"| TOOL_C
    CA -->|"trusted by"| REC_C

    style CA fill:#DC3545,stroke:#A71D2A,color:#fff
    style SC fill:#FFC107,stroke:#CC9A06,color:#000
```

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
| Planning fails | Fall back to unguided single ReACT loop |
| Reflection parse fails | Skip refinement, use original response |
| Reflexion store fails | Log warning, continue (non-blocking) |
| Evaluation store fails | Log warning, continue (non-blocking) |

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

| Setting                       | Default | Description                             |
| ----------------------------- | ------- | --------------------------------------- |
| `REACT_MAX_ITERATIONS`        | `8`     | Maximum ReACT reasoning steps per query |
| `REACT_TIMEOUT_SECONDS`       | `120`   | Total pipeline timeout in seconds       |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.8`   | Below this, ask for clarification       |

### Reflection (Stage 8)

| Setting                        | Default | Description                               |
| ------------------------------ | ------- | ----------------------------------------- |
| `REFLECTION_ENABLED`           | `true`  | Enable post-response quality evaluation   |
| `REFLECTION_MAX_ITERATIONS`    | `2`     | Max evaluate-refine cycles                |
| `REFLECTION_QUALITY_THRESHOLD` | `0.75`  | Score above which refinement stops        |
| `REFLECTION_TEMPERATURE`       | `0.2`   | LLM temperature for evaluation/refinement |
| `REFLECTION_MAX_TOKENS`        | `512`   | Max tokens for evaluation/refinement      |

### Reflexion (Stage 10)

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

When all pipeline features are disabled (`PLANNING_ENABLED=false`, `MULTI_AGENT_ENABLED=false`, `GUARDRAILS_ENABLED=false`, `REFLECTION_ENABLED=false`, `REFLEXION_ENABLED=false`, `EVALUATION_STORAGE_ENABLED=false`), the system operates as a basic ReACT agent with intent classification and tool use.

---

## Event Reference

All events streamed to the client via WebSocket during query processing:

| Event Type              | Stage | Payload                                      | Description                      |
| ----------------------- | ----- | -------------------------------------------- | -------------------------------- |
| `processing_started`    | 0     | `{}`                                         | Query processing began           |
| `guardrail_blocked`     | 1     | `{reason, type}`                             | Input blocked by safety filter   |
| `clarification`         | 3     | `{message, options, allow_freetext}`         | Asking user for more context     |
| `agent_planning`        | 4     | `{steps[], multi_agent}`                     | Plan decomposition result        |
| `agent_started`         | 5     | `{agent_type, description}`                  | Specialist agent launched        |
| `agent_thinking`        | 5     | `{iteration, thought, action, has_answer}`   | ReACT reasoning step             |
| `tool_validation_error` | 5     | `{tool, error}`                              | Tool parameter validation failed |
| `agent_complete`        | 5     | `{agent_type, tools_used[]}`                 | Specialist agent finished        |
| `reflection_evaluating` | 8     | `{iteration}`                                | Evaluating response quality      |
| `reflection_critique`   | 8     | `{score, issues[], needs_refinement}`        | Quality evaluation result        |
| `reflection_refining`   | 8     | `{iteration}`                                | Improving response               |
| `guardrail_sanitized`   | 9     | `{redactions[]}`                             | PII redacted from output         |
| `reflexion_learning`    | 10    | `{message}`                                  | Learning from poor interaction   |
| `token`                 | 12    | `{token}`                                    | Streamed response token          |
| `response_complete`     | 12    | `{text, confidence, sources, suggestions[]}` | Final response with metadata     |
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
