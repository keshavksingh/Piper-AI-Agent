"""Agent Service — ReACT loop, intent classification, clarification, response framing,
reflection (post-response evaluate/refine), reflexion (persistent learning), and
structured tool validation."""

import json
import re
import sys
import time
import threading
import grpc
from concurrent import futures

sys.path.append("..")

import protos.agent_service_pb2 as agent_pb2
import protos.agent_service_pb2_grpc as agent_pb2_grpc
import protos.memory_service_pb2 as memory_pb2
import protos.memory_service_pb2_grpc as memory_pb2_grpc
import protos.llm_service_pb2 as llm_pb2
import protos.llm_service_pb2_grpc as llm_pb2_grpc
import protos.tool_service_pb2 as tool_pb2
import protos.tool_service_pb2_grpc as tool_pb2_grpc
import protos.recommendation_service_pb2 as rec_pb2
import protos.recommendation_service_pb2_grpc as rec_pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging
from shared.resilience import create_grpc_channel, grpc_retry

log = setup_logging("agent_service")

# ── gRPC Stubs (cached channels to prevent resource leaks) ───────

_channel_cache = {}
_channel_lock = threading.Lock()


def _get_cached_channel(addr):
    """Return a cached gRPC channel for the given address, creating one if needed."""
    if addr in _channel_cache:
        return _channel_cache[addr]
    with _channel_lock:
        # Double-check after acquiring lock
        if addr not in _channel_cache:
            _channel_cache[addr] = create_grpc_channel(addr)
        return _channel_cache[addr]


def get_memory_stub():
    channel = _get_cached_channel(Config.MEMORY_SERVICE_ADDR)
    return memory_pb2_grpc.MemoryServiceStub(channel)

def get_llm_stub():
    channel = _get_cached_channel(Config.LLM_SERVICE_ADDR)
    return llm_pb2_grpc.LLMServiceStub(channel)

def get_tool_stub():
    channel = _get_cached_channel(Config.TOOL_SERVICE_ADDR)
    return tool_pb2_grpc.ToolServiceStub(channel)

def get_rec_stub():
    channel = _get_cached_channel(Config.RECOMMENDATION_SERVICE_ADDR)
    return rec_pb2_grpc.RecommendationServiceStub(channel)


# ── Prompt Templates ──────────────────────────────────────────────

INTENT_CLASSIFICATION_PROMPT = """Classify the user's intent based on their query and conversation context.

Conversation context:
{context}

User query: {query}

Respond with valid JSON only, no markdown:
{{
  "intent": "product_inquiry|price_check|comparison|warranty_question|general_question|follow_up|out_of_scope",
  "confidence": 0.0-1.0,
  "entities": ["extracted product names or attributes"],
  "needs_clarification": true or false,
  "clarification_question": "question to ask if ambiguous (empty string if not needed)"
}}"""

REACT_SYSTEM_PROMPT = """You are Piper, an AI customer support agent for a product catalog. You help users find products, compare prices, check warranties, and answer product questions.

You must reason step-by-step using the ReACT framework. In each step, you produce a Thought and then either an Action or a Final Answer.
{reflexion_context}
Available tools:
{tool_descriptions}

Rules:
- Always think before acting
- Use tools to get factual information; do not make up product details
- If you have enough information, provide a Final Answer
- Be concise in your thoughts
- Format your response as exactly one of these two patterns:

Pattern 1 (need more info):
Thought: [your reasoning about what you need to find out]
Action: tool_name({{"param": "value"}})

Pattern 2 (ready to answer):
Thought: [your reasoning about why you can now answer]
Answer: [your final response to the user]"""

REACT_USER_PROMPT = """Session context:
{memory_context}

User query: {query}

{react_history}

Continue the reasoning. Respond with a Thought and either an Action or an Answer."""

RESPONSE_FRAMING_PROMPT = """You are framing a final response for the user based on the agent's reasoning.

Original query: {query}
Agent's answer: {answer}
Tools used: {tools_used}
Reasoning steps: {reasoning_steps}

Create a helpful, natural response. Include the factual information found.
Respond with valid JSON only, no markdown:
{{
  "text": "your polished response to the user",
  "confidence": 0.0-1.0,
  "sources": ["list of product names or data sources used"]
}}"""

# ── Reflection Prompt Templates ───────────────────────────────────

REFLECTION_EVALUATE_PROMPT = """Evaluate the quality of this customer support response.

User query: {query}
Response: {response_text}
Tools used: {tools_used}
Number of reasoning steps: {num_steps}
Memory context: {memory_context}

Score each criterion from 0.0 to 1.0:
- completeness: Does the response fully address the user's question?
- accuracy: Is the information factually correct based on tool results?
- relevance: Is the response focused on what the user asked?
- clarity: Is the response clear and easy to understand?
- actionability: Does the response give the user useful next steps?

Respond with valid JSON only, no markdown:
{{
  "completeness": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "relevance": 0.0-1.0,
  "clarity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "issues": ["list of specific issues found"],
  "suggestions": ["list of specific improvements"],
  "needs_refinement": true or false
}}"""

REFLECTION_REFINE_PROMPT = """Improve this customer support response based on the critique.

User query: {query}
Original response: {response_text}
Critique: {critique}
Tool observations: {observations}

Produce an improved response that addresses the identified issues.
Respond with valid JSON only, no markdown:
{{
  "text": "your improved response to the user",
  "confidence": 0.0-1.0,
  "sources": ["list of product names or data sources used"]
}}"""

# ── Reflexion Prompt Templates ────────────────────────────────────

REFLEXION_SELF_REFLECT_PROMPT = """Analyze this interaction to extract a reusable learning.

User query: {query}
Intent: {intent}
Tools used: {tools_used}
Original response quality score: {original_score}
Issues found: {issues}
Refined response quality score: {refined_score}

Generate a concise learning that can help handle similar queries better in the future.
Respond with valid JSON only, no markdown:
{{
  "query_pattern": "a general description of the type of query",
  "failure_reason": "what went wrong with the original response",
  "suggested_improvement": "what to do differently next time",
  "key_topics": ["relevant topics for matching future queries"]
}}"""

# ── Planning Prompt Template ─────────────────────────────────────

PLANNING_PROMPT = """Analyze the user's query and create an execution plan.

User query: {query}
Detected intent: {intent}
Available tools: {tool_list}
Conversation context: {context}

Respond with valid JSON only:
{{
  "needs_multi_agent": <true if query involves comparison/multiple products/complex multi-step>,
  "plan_steps": [
    {{"goal": "<sub-goal description>", "suggested_tool": "<tool_name or null>", "priority": <1-3>}}
  ],
  "specialist_agents": ["<agent_type>"]
}}"""

# ── Agent Registry ───────────────────────────────────────────────

AGENT_REGISTRY = {
    "product_specialist": {
        "description": "Expert on product details, features, and specifications",
        "system_prompt_suffix": "Focus on product details, specifications, and features. Be thorough about technical specs.",
        "preferred_tools": ["product_search", "price_lookup"],
    },
    "warranty_specialist": {
        "description": "Expert on warranty policies, claims, and coverage",
        "system_prompt_suffix": "Focus on warranty information, coverage terms, and claim procedures.",
        "preferred_tools": ["warranty_check"],
    },
    "comparison_specialist": {
        "description": "Expert at comparing products across dimensions",
        "system_prompt_suffix": "Compare products systematically across features, price, and value. Use tables when helpful.",
        "preferred_tools": ["product_search", "price_lookup", "product_compare"],
    },
}

# ── Guardrail Patterns ──────────────────────────────────────────

_PII_PATTERNS = [
    ("email", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
    ("phone", re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')),
    ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("credit_card", re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')),
]

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+', re.IGNORECASE),
    re.compile(r'system\s*:\s*', re.IGNORECASE),
    re.compile(r'<\s*(?:system|admin|root)\s*>', re.IGNORECASE),
    re.compile(r'(?:forget|disregard)\s+(?:everything|all|your)', re.IGNORECASE),
]

_PII_REDACTION_MAP = {
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE REDACTED]",
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CARD REDACTED]",
}

# ── ReACT Loop ────────────────────────────────────────────────────

def parse_react_output(text: str):
    """Parse LLM output into thought, action/answer components."""
    thought = ""
    action = None
    action_input = None
    answer = None

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nAnswer:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Check for Action — use greedy match inside parens to handle nested JSON
    action_match = re.search(r"Action:\s*(\w+)\((.+)\)\s*$", text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        raw_input = action_match.group(2).strip()
        try:
            json.loads(raw_input)
            action_input = raw_input
        except (json.JSONDecodeError, ValueError):
            # Keep raw input — downstream tool validation will handle it
            action_input = raw_input

    # Check for Answer
    answer_match = re.search(r"Answer:\s*(.+?)$", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return thought, action, action_input, answer


def build_react_history(steps: list) -> str:
    """Build the history string from previous ReACT steps."""
    if not steps:
        return "No previous reasoning steps."

    history = "Previous reasoning:\n"
    for step in steps:
        history += f"Thought {step['iteration']}: {step['thought']}\n"
        if step.get("action"):
            history += f"Action {step['iteration']}: {step['action']}({step.get('action_input', '{}')})\n"
            history += f"Observation {step['iteration']}: {step.get('observation', 'N/A')}\n"
    return history


def build_memory_context(turns) -> str:
    """Build context string from conversation turns."""
    if not turns:
        return "No previous conversation."

    lines = []
    for turn in turns[-10:]:  # Last 10 turns
        role_label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{role_label}: {turn.content}")
    return "\n".join(lines)


def _strip_llm_json(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


# ── Service Implementation ────────────────────────────────────────

class AgentServiceServicer(agent_pb2_grpc.AgentServiceServicer):

    @grpc_retry
    def _classify_intent(self, query: str, context: str):
        """Classify user intent using the LLM."""
        llm_stub = get_llm_stub()

        prompt = INTENT_CLASSIFICATION_PROMPT.format(context=context, query=query)
        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are an intent classifier. Respond with valid JSON only.",
            temperature=Config.INTENT_TEMPERATURE,
            max_tokens=Config.INTENT_MAX_TOKENS,
        ))

        try:
            result = json.loads(_strip_llm_json(response.completion))
            return result
        except json.JSONDecodeError:
            log.warning("intent_parse_failed", raw=response.completion)
            return {
                "intent": "general_question",
                "confidence": 0.5,
                "entities": [],
                "needs_clarification": False,
                "clarification_question": "",
            }

    # ── Tool Descriptions & Schemas ──────────────────────────────

    @grpc_retry
    def _get_tool_descriptions(self):
        """Fetch available tools from Tool Service, including parameter schemas."""
        tool_stub = get_tool_stub()
        response = tool_stub.ListTools(tool_pb2.ListToolsRequest())
        descriptions = []
        tool_schemas = {}
        for tool in response.tools:
            schema_str = tool.parameter_schema
            descriptions.append(
                f"- {tool.name}: {tool.description}\n  Parameters: {schema_str}"
            )
            try:
                tool_schemas[tool.name] = json.loads(schema_str)
            except (json.JSONDecodeError, TypeError):
                tool_schemas[tool.name] = {}
        return "\n".join(descriptions), [t.name for t in response.tools], tool_schemas

    # ── Structured Tool Validation ───────────────────────────────

    def _validate_tool_params(self, tool_name, params_json, schema):
        """Validate tool call parameters against their schema.

        Returns (is_valid, parsed_params, error_message).
        Lightweight checks — no jsonschema dependency.
        """
        if not Config.TOOL_VALIDATION_ENABLED:
            return True, params_json, ""

        # Check JSON parseability
        try:
            params = json.loads(params_json) if isinstance(params_json, str) else params_json
        except (json.JSONDecodeError, TypeError):
            return False, params_json, f"Invalid JSON for {tool_name} parameters: {params_json}"

        if not schema:
            return True, json.dumps(params) if not isinstance(params_json, str) else params_json, ""

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in params:
                return False, params_json, f"Missing required field '{field}' for tool {tool_name}"
            if params[field] is None or (isinstance(params[field], str) and not params[field].strip()):
                return False, params_json, f"Required field '{field}' is empty for tool {tool_name}"

        # Basic type checks
        for field, value in params.items():
            if field not in properties:
                continue
            expected_type = properties[field].get("type", "")
            if expected_type == "string" and not isinstance(value, str):
                return False, params_json, f"Field '{field}' should be a string for tool {tool_name}, got {type(value).__name__}"
            if expected_type == "integer" and not isinstance(value, int):
                return False, params_json, f"Field '{field}' should be an integer for tool {tool_name}, got {type(value).__name__}"
            if expected_type == "number" and not isinstance(value, (int, float)):
                return False, params_json, f"Field '{field}' should be a number for tool {tool_name}, got {type(value).__name__}"
            if expected_type == "array" and not isinstance(value, list):
                return False, params_json, f"Field '{field}' should be an array for tool {tool_name}, got {type(value).__name__}"

        return True, params_json, ""

    def _validate_tool_result(self, tool_name, result_json):
        """Validate tool output and enrich observation if issues found.

        Returns (is_valid, enriched_observation).
        """
        if not Config.TOOL_VALIDATION_ENABLED:
            return True, result_json

        try:
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
        except (json.JSONDecodeError, TypeError):
            # Non-JSON result — pass through as-is
            return True, result_json

        # Check for error key in result
        if isinstance(result, dict) and "error" in result:
            enriched = f"{result_json}\nNote: The tool returned an error. Consider trying with different parameters."
            return False, enriched

        # Per-tool result completeness checks
        if isinstance(result, dict):
            items = result.get("results", result.get("products", result.get("items", None)))
            if isinstance(items, list) and len(items) == 0:
                enriched = f"{result_json}\nNote: No results found. Consider broadening your search or using different terms."
                return False, enriched

        if isinstance(result, list) and len(result) == 0:
            enriched = f"{result_json}\nNote: No results found. Consider broadening your search or using different terms."
            return False, enriched

        return True, result_json

    # ── ReACT Step Execution ─────────────────────────────────────

    @grpc_retry
    def _execute_react_step(self, query, memory_context, tool_descriptions, react_history,
                            reflexion_context=""):
        """Execute one iteration of the ReACT loop via LLM."""
        llm_stub = get_llm_stub()

        system = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            reflexion_context=reflexion_context,
        )
        user = REACT_USER_PROMPT.format(
            memory_context=memory_context,
            query=query,
            react_history=react_history,
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=user,
            system_prompt=system,
            temperature=Config.DEFAULT_TEMPERATURE,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
        ))

        return response.completion

    @grpc_retry
    def _execute_tool(self, session_id, tool_name, params_json):
        """Execute a tool via Tool Service."""
        tool_stub = get_tool_stub()
        response = tool_stub.ExecuteTool(tool_pb2.ExecuteToolRequest(
            session_id=session_id,
            tool_name=tool_name,
            parameters=params_json,
        ))
        return response

    @grpc_retry
    def _frame_response(self, query, answer, tools_used, reasoning_steps):
        """Frame the final response with confidence and sources."""
        llm_stub = get_llm_stub()

        prompt = RESPONSE_FRAMING_PROMPT.format(
            query=query,
            answer=answer,
            tools_used=", ".join(tools_used) if tools_used else "none",
            reasoning_steps=reasoning_steps,
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a response formatter. Respond with valid JSON only.",
            temperature=0.2,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
        ))

        try:
            return json.loads(_strip_llm_json(response.completion))
        except json.JSONDecodeError:
            return {
                "text": answer,
                "confidence": 0.7,
                "sources": [],
            }

    # ── Reflection Agent ─────────────────────────────────────────

    @grpc_retry
    def _evaluate_response(self, query, response_text, tools_used, num_steps, memory_context):
        """LLM call to score response quality on multiple criteria."""
        llm_stub = get_llm_stub()

        prompt = REFLECTION_EVALUATE_PROMPT.format(
            query=query,
            response_text=response_text,
            tools_used=", ".join(tools_used) if tools_used else "none",
            num_steps=num_steps,
            memory_context=memory_context[:500],
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a quality evaluator. Respond with valid JSON only.",
            temperature=Config.REFLECTION_TEMPERATURE,
            max_tokens=Config.REFLECTION_MAX_TOKENS,
        ))

        try:
            return json.loads(_strip_llm_json(response.completion))
        except json.JSONDecodeError:
            log.warning("reflection_evaluate_parse_failed", raw=response.completion[:200])
            # Assume score=1.0 to skip refinement on parse failure
            return {
                "completeness": 1.0, "accuracy": 1.0, "relevance": 1.0,
                "clarity": 1.0, "actionability": 1.0,
                "overall_score": 1.0, "issues": [], "suggestions": [],
                "needs_refinement": False,
            }

    @grpc_retry
    def _refine_response(self, query, response_text, tools_used, evaluation, observations):
        """LLM call to produce an improved response based on critique."""
        llm_stub = get_llm_stub()

        prompt = REFLECTION_REFINE_PROMPT.format(
            query=query,
            response_text=response_text,
            critique=json.dumps({
                "overall_score": evaluation.get("overall_score", 0),
                "issues": evaluation.get("issues", []),
                "suggestions": evaluation.get("suggestions", []),
            }),
            observations=observations[:1000],
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a response refiner. Respond with valid JSON only.",
            temperature=Config.REFLECTION_TEMPERATURE,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
        ))

        try:
            return json.loads(_strip_llm_json(response.completion))
        except json.JSONDecodeError:
            log.warning("reflection_refine_parse_failed", raw=response.completion[:200])
            return None

    def _run_reflection_loop(self, query, framed, tools_used, steps, memory_context):
        """Orchestrate evaluate -> refine loop. Returns (refined_framed_dict, list_of_AgentEvents, original_score, last_evaluation)."""
        events = []
        current = framed
        original_score = None
        last_evaluation = {}

        observations = "\n".join(
            f"- {s.get('action', 'N/A')}: {s.get('observation', 'N/A')[:200]}"
            for s in steps if s.get("observation")
        )

        for i in range(1, Config.REFLECTION_MAX_ITERATIONS + 1):
            # Emit evaluating event
            events.append(agent_pb2.AgentEvent(
                type="reflection_evaluating",
                payload=json.dumps({"iteration": i, "step": "Evaluating response quality..."}),
            ))

            evaluation = self._evaluate_response(
                query, current["text"], tools_used, len(steps), memory_context
            )
            last_evaluation = evaluation
            score = evaluation.get("overall_score", 1.0)
            issues = evaluation.get("issues", [])

            if original_score is None:
                original_score = score

            log.info("reflection_evaluation", iteration=i, score=score, issues=issues)

            # Emit critique event
            events.append(agent_pb2.AgentEvent(
                type="reflection_critique",
                payload=json.dumps({
                    "iteration": i,
                    "score": score,
                    "issues": issues,
                    "suggestions": evaluation.get("suggestions", []),
                }),
            ))

            # Good enough — stop
            if score >= Config.REFLECTION_QUALITY_THRESHOLD:
                log.info("reflection_passed", iteration=i, score=score)
                break

            # Evaluator says no refinement needed
            if not evaluation.get("needs_refinement", False):
                log.info("reflection_no_refinement_needed", iteration=i, score=score)
                break

            # Emit refining event
            events.append(agent_pb2.AgentEvent(
                type="reflection_refining",
                payload=json.dumps({"iteration": i, "step": "Refining response..."}),
            ))

            refined = self._refine_response(
                query, current["text"], tools_used, evaluation, observations
            )

            if refined is not None:
                current = refined
                log.info("reflection_refined", iteration=i)
            else:
                # Parse failure — keep previous response
                log.warning("reflection_refine_failed_keeping_previous", iteration=i)
                break

        return current, events, original_score, last_evaluation

    # ── Reflexion Agent (Persistent Learning) ────────────────────

    def _get_reflexion_insights(self, customer_id, intent, query):
        """Fetch past reflexion insights from episodic memory.

        Returns a formatted string for injection into the ReACT system prompt.
        """
        if not Config.REFLEXION_ENABLED or not customer_id:
            return ""

        try:
            memory_stub = get_memory_stub()
            response = memory_stub.GetEpisodicMemories(memory_pb2.GetEpisodicRequest(
                customer_id=customer_id,
                limit=5,
                event_type="reflexion_insight",
            ))

            if not response.memories:
                return ""

            # Filter by intent or key_topics overlap with query terms
            query_terms = set(query.lower().split())
            relevant = []
            for mem in response.memories:
                # Match by intent in metadata
                try:
                    meta = json.loads(mem.metadata) if mem.metadata else {}
                except json.JSONDecodeError:
                    meta = {}

                mem_intent = meta.get("intent", "")
                mem_topics = set(t.lower() for t in mem.key_topics)

                # Relevance: intent match or topic overlap
                if mem_intent == intent or query_terms & mem_topics:
                    relevant.append(mem)

            if not relevant:
                # Fall back to most recent insights
                relevant = list(response.memories)[:Config.REFLEXION_MAX_INSIGHTS_PER_QUERY]

            # Format for prompt injection
            insights = relevant[:Config.REFLEXION_MAX_INSIGHTS_PER_QUERY]
            lines = ["\nLearnings from past interactions (use these to improve your response):"]
            for ins in insights:
                lines.append(f"- {ins.summary}")
            lines.append("")

            context = "\n".join(lines)
            log.info("reflexion_insights_loaded", count=len(insights))
            return context

        except Exception as e:
            log.warning("reflexion_insights_fetch_failed", error=str(e))
            return ""

    @grpc_retry
    def _generate_reflexion_insight(self, query, intent, tools_used, evaluation, original_score,
                                     refined_score):
        """LLM call to produce a reusable learning from a poor interaction."""
        llm_stub = get_llm_stub()

        prompt = REFLEXION_SELF_REFLECT_PROMPT.format(
            query=query,
            intent=intent,
            tools_used=", ".join(tools_used) if tools_used else "none",
            original_score=original_score,
            issues=json.dumps(evaluation.get("issues", [])),
            refined_score=refined_score,
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a self-reflection agent. Respond with valid JSON only.",
            temperature=Config.REFLECTION_TEMPERATURE,
            max_tokens=Config.REFLECTION_MAX_TOKENS,
        ))

        try:
            return json.loads(_strip_llm_json(response.completion))
        except json.JSONDecodeError:
            log.warning("reflexion_insight_parse_failed", raw=response.completion[:200])
            return None

    def _maybe_store_reflexion_insight(self, session_id, customer_id, query, intent,
                                        tools_used, evaluation, original_score, refined_score,
                                        framed_text):
        """Store a reflexion insight if the original quality was below threshold.

        Returns a list of AgentEvents (0 or 1).
        Non-blocking: failures are logged and swallowed.
        """
        events = []
        if not Config.REFLEXION_ENABLED:
            return events
        if original_score is None or original_score >= Config.REFLEXION_INSIGHT_THRESHOLD:
            return events

        try:
            insight = self._generate_reflexion_insight(
                query, intent, tools_used, evaluation, original_score, refined_score
            )
            if insight is None:
                return events

            memory_stub = get_memory_stub()
            summary = insight.get("suggested_improvement", "")
            if not summary:
                summary = insight.get("failure_reason", "Unspecified learning")

            metadata = json.dumps({
                "query_pattern": insight.get("query_pattern", ""),
                "intent": intent,
                "failure_reason": insight.get("failure_reason", ""),
                "suggested_improvement": insight.get("suggested_improvement", ""),
                "original_score": original_score,
                "refined_score": refined_score,
                "tools_used": tools_used,
            })

            refinement_helped = (refined_score or 0) > (original_score or 0)

            memory_stub.StoreEpisodicMemory(memory_pb2.StoreEpisodicRequest(
                customer_id=customer_id,
                session_id=session_id,
                event_type="reflexion_insight",
                summary=summary,
                key_topics=insight.get("key_topics", []),
                resolution_status="resolved" if refinement_helped else "unresolved",
                metadata=metadata,
            ))

            log.info("reflexion_insight_stored",
                     original_score=original_score,
                     refined_score=refined_score)

            events.append(agent_pb2.AgentEvent(
                type="reflexion_learning",
                payload=json.dumps({"message": "Learning from this interaction..."}),
            ))

        except Exception as e:
            log.warning("reflexion_insight_store_failed", error=str(e))

        return events

    # ── Planning Layer ────────────────────────────────────────────

    @grpc_retry
    def _generate_plan(self, query, intent, tool_list, context):
        """LLM call to decompose query into sub-goals."""
        llm_stub = get_llm_stub()

        prompt = PLANNING_PROMPT.format(
            query=query,
            intent=intent,
            tool_list=tool_list,
            context=context[:500],
        )

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a planning agent. Respond with valid JSON only.",
            temperature=Config.PLANNING_TEMPERATURE,
            max_tokens=Config.PLANNING_MAX_TOKENS,
        ))

        try:
            plan = json.loads(_strip_llm_json(response.completion))
            log.info("plan_generated",
                     steps=len(plan.get("plan_steps", [])),
                     multi_agent=plan.get("needs_multi_agent", False))
            return plan
        except json.JSONDecodeError:
            log.warning("plan_parse_failed", raw=response.completion[:200])
            return {"needs_multi_agent": False, "plan_steps": [], "specialist_agents": []}

    # ── Input/Output Guardrails ──────────────────────────────────

    def _check_input_guardrails(self, query):
        """Regex PII + injection detection on input.

        Returns (is_safe, sanitized_query, issues).
        """
        issues = []

        # Length check
        if len(query) > Config.GUARDRAILS_MAX_QUERY_LENGTH:
            return False, query, [{"type": "length_exceeded",
                                   "detail": f"Query exceeds maximum length of {Config.GUARDRAILS_MAX_QUERY_LENGTH} characters"}]

        # Prompt injection detection — blocks
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                return False, query, [{"type": "injection",
                                       "detail": "Potential prompt injection detected. Please rephrase your query."}]

        # PII detection — warns but does not block
        detected_pii = []
        for pii_type, pattern in _PII_PATTERNS:
            if pattern.search(query):
                detected_pii.append(pii_type)

        if detected_pii:
            issues.append({"type": "pii_warning", "pii_types": detected_pii})

        return True, query, issues

    def _check_output_guardrails(self, response_text):
        """PII redaction on output.

        Returns (sanitized_text, was_modified, redactions).
        """
        sanitized = response_text
        redactions = []

        for pii_type, pattern in _PII_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub(_PII_REDACTION_MAP[pii_type], sanitized)
                redactions.append(pii_type)

        return sanitized, len(redactions) > 0, redactions

    # ── Evaluation Storage ───────────────────────────────────────

    def _store_evaluation_record(self, session_id, customer_id, query, intent,
                                  response_text, tools_used, steps, confidence,
                                  reflection_score, latency_seconds):
        """Store structured evaluation data to TimescaleDB via episodic memory."""
        memory_stub = get_memory_stub()

        payload = json.dumps({
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "reflection_score": reflection_score,
            "tools_used": tools_used,
            "reasoning_steps": len(steps),
            "latency_ms": int(latency_seconds * 1000),
            "response_length": len(response_text),
        })

        memory_stub.StoreEpisodicMemory(memory_pb2.StoreEpisodicRequest(
            customer_id=customer_id,
            session_id=session_id,
            event_type="evaluation_record",
            summary=f"Evaluation: intent={intent}, confidence={confidence:.2f}, latency={int(latency_seconds * 1000)}ms",
            key_topics=[intent] + tools_used,
            resolution_status="resolved",
            metadata=payload,
        ))

        log.info("evaluation_record_stored",
                 intent=intent, confidence=confidence,
                 latency_ms=int(latency_seconds * 1000))

    # ── Multi-Agent Orchestration ────────────────────────────────

    def _run_agent_sub_loop(self, agent_type, query, memory_context,
                             tool_descriptions, available_tools, tool_schemas):
        """Run a focused ReACT loop (max 4 iterations) for a single specialist agent."""
        agent_config = AGENT_REGISTRY.get(agent_type, {})
        suffix = agent_config.get("system_prompt_suffix", "")

        # Bias tool descriptions toward preferred tools
        preferred = agent_config.get("preferred_tools", [])

        steps = []
        tools_used = []
        final_answer = None
        max_sub_iterations = 4

        for iteration in range(1, max_sub_iterations + 1):
            react_history = build_react_history(steps)

            # Build a specialized system prompt
            system = REACT_SYSTEM_PROMPT.format(
                tool_descriptions=tool_descriptions,
                reflexion_context=f"\nSpecialist focus: {suffix}\n" if suffix else "",
            )
            user = REACT_USER_PROMPT.format(
                memory_context=memory_context,
                query=query,
                react_history=react_history,
            )

            llm_stub = get_llm_stub()
            response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
                prompt=user,
                system_prompt=system,
                temperature=Config.DEFAULT_TEMPERATURE,
                max_tokens=Config.DEFAULT_MAX_TOKENS,
            ))

            thought, action, action_input, answer = parse_react_output(response.completion)

            if answer:
                final_answer = answer
                steps.append({"iteration": iteration, "thought": thought})
                break

            if action and action in available_tools:
                # ── Tool Parameter Validation ─────────────────────
                if Config.TOOL_VALIDATION_ENABLED:
                    schema = tool_schemas.get(action, {})
                    is_valid, validated_params, error_msg = self._validate_tool_params(
                        action, action_input or "{}", schema
                    )
                    if not is_valid:
                        log.warning("sub_loop_tool_param_validation_failed",
                                    tool=action, error=error_msg)
                        steps.append({
                            "iteration": iteration,
                            "thought": thought,
                            "action": action,
                            "action_input": action_input,
                            "observation": f"Parameter validation error: {error_msg}. Please fix the parameters and try again.",
                        })
                        continue

                try:
                    tool_response = self._execute_tool("sub_loop", action, action_input or "{}")
                    observation = tool_response.result if tool_response.success else f"Error: {tool_response.error}"
                    tools_used.append(action)

                    # ── Tool Result Validation ────────────────────
                    if Config.TOOL_VALIDATION_ENABLED and tool_response.success:
                        _, enriched_obs = self._validate_tool_result(action, observation)
                        observation = enriched_obs

                except Exception as e:
                    observation = f"Tool execution failed: {str(e)}"

                steps.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })
            else:
                final_answer = response.completion
                break

        if final_answer is None:
            observations = "\n".join(
                f"- {s.get('action', 'N/A')}: {s.get('observation', 'N/A')[:200]}"
                for s in steps if s.get("observation")
            )
            final_answer = observations or "No findings."

        return final_answer, tools_used, steps

    @grpc_retry
    def _synthesize_multi_agent_response(self, results, query):
        """LLM call to combine specialist outputs into a coherent answer."""
        llm_stub = get_llm_stub()

        parts = []
        for agent_type, result_text, tools_used in results:
            parts.append(f"[{agent_type}]:\n{result_text}")

        combined = "\n\n".join(parts)

        prompt = f"""Synthesize these specialist agent outputs into a single coherent response for the user.

Original query: {query}

Specialist outputs:
{combined}

Provide a unified, well-structured response that combines all the specialist findings. Be concise and helpful."""

        response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
            prompt=prompt,
            system_prompt="You are a response synthesizer. Combine specialist outputs into a clear, unified answer.",
            temperature=0.3,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
        ))

        return response.completion

    def _run_multi_agent_loop(self, session_id, customer_id, query, intent,
                               memory_context, memory_stub, start_time, plan,
                               tool_descriptions, available_tools, tool_schemas):
        """Orchestrate sequential specialist agents and synthesize results."""
        specialist_agents = plan.get("specialist_agents", [])[:Config.MULTI_AGENT_MAX_AGENTS]

        # Filter to known agents
        specialist_agents = [a for a in specialist_agents if a in AGENT_REGISTRY]
        if not specialist_agents:
            # Fall back to single ReACT loop
            yield from self._run_react_loop(
                session_id, customer_id, query, intent,
                memory_context, memory_stub, start_time,
                plan=plan, tool_descriptions=tool_descriptions,
                available_tools=available_tools, tool_schemas=tool_schemas,
            )
            return

        results = []
        all_tools_used = []

        for agent_type in specialist_agents:
            agent_config = AGENT_REGISTRY[agent_type]

            # Emit agent_started
            yield agent_pb2.AgentEvent(
                type="agent_started",
                payload=json.dumps({
                    "agent_type": agent_type,
                    "description": agent_config["description"],
                }),
            )

            result_text, tools_used, steps = self._run_agent_sub_loop(
                agent_type, query, memory_context,
                tool_descriptions, available_tools, tool_schemas,
            )

            results.append((agent_type, result_text, tools_used))
            all_tools_used.extend(tools_used)

            # Emit agent_complete
            yield agent_pb2.AgentEvent(
                type="agent_complete",
                payload=json.dumps({
                    "agent_type": agent_type,
                    "tools_used": tools_used,
                }),
            )

        # Synthesize all specialist outputs
        try:
            synthesized = self._synthesize_multi_agent_response(results, query)
        except Exception as e:
            log.error("synthesis_failed", error=str(e))
            synthesized = "\n\n".join(f"{r[0]}: {r[1]}" for r in results)

        # Frame the synthesized response
        all_steps = []  # Combined steps placeholder
        try:
            framed = self._frame_response(query, synthesized, all_tools_used, len(results))
        except Exception:
            framed = {"text": synthesized, "confidence": 0.7, "sources": []}

        # Reflection on synthesized response
        original_score = None
        refined_score = None

        if Config.REFLECTION_ENABLED:
            framed, reflection_events, original_score, _ = self._run_reflection_loop(
                query, framed, all_tools_used, all_steps, memory_context
            )
            for event in reflection_events:
                yield event
            refined_score = framed.get("confidence", original_score)

        # Output guardrails
        if Config.GUARDRAILS_ENABLED:
            sanitized_text, was_modified, redactions = self._check_output_guardrails(framed["text"])
            if was_modified:
                framed["text"] = sanitized_text
                log.warning("output_pii_redacted", redactions=redactions)
                yield agent_pb2.AgentEvent(
                    type="guardrail_sanitized",
                    payload=json.dumps({"redacted_types": redactions}),
                )

        # Stream tokens
        for word in framed["text"].split(" "):
            yield agent_pb2.AgentEvent(type="token", payload=word + " ")

        # Store assistant turn
        try:
            memory_stub.AddConversationTurn(memory_pb2.AddTurnRequest(
                session_id=session_id,
                role="assistant",
                content=framed["text"],
                intent=intent,
                confidence=framed.get("confidence", 0.7),
                tool_calls=json.dumps([{"tool": t} for t in all_tools_used]),
            ))
        except Exception as e:
            log.warning("store_assistant_turn_failed", error=str(e))

        # Evaluation storage
        if Config.EVALUATION_STORAGE_ENABLED:
            try:
                elapsed = time.time() - start_time
                self._store_evaluation_record(
                    session_id, customer_id, query, intent,
                    framed["text"], all_tools_used, all_steps,
                    framed.get("confidence", 0.7),
                    original_score, elapsed,
                )
            except Exception as e:
                log.warning("evaluation_storage_failed", error=str(e))

        # Get recommendations
        recommendations = []
        try:
            rec_stub = get_rec_stub()
            rec_response = rec_stub.GetFollowUpRecommendations(
                rec_pb2.FollowUpRecommendationRequest(
                    session_id=session_id,
                    last_query=query,
                    last_response=framed["text"][:500],
                    intent=intent,
                    customer_id=customer_id,
                )
            )
            recommendations = list(rec_response.suggestions)
        except Exception as e:
            log.warning("get_recommendations_failed", error=str(e))

        # Send response_complete
        complete_payload = json.dumps({
            "response": {
                "text": framed["text"],
                "confidence": framed.get("confidence", 0.7),
                "sources": framed.get("sources", []),
                "reasoning_steps": len(results),
                "tools_used": all_tools_used,
            },
            "recommendations": recommendations,
        })
        yield agent_pb2.AgentEvent(type="response_complete", payload=complete_payload)

    # ── Main Entry Points ────────────────────────────────────────

    def ProcessQuery(self, request, context):
        """Main entry point: process a user query through the ReACT loop."""
        session_id = request.session_id
        customer_id = request.customer_id
        query = request.query

        log.info("query_received", session_id=session_id, query=query)
        start_time = time.time()

        memory_stub = get_memory_stub()

        # Touch session (refresh TTL)
        try:
            memory_stub.TouchSession(memory_pb2.TouchSessionRequest(session_id=session_id))
        except Exception as e:
            log.warning("touch_session_failed", error=str(e))

        # Get conversation history
        try:
            history_resp = memory_stub.GetConversationHistory(
                memory_pb2.GetHistoryRequest(session_id=session_id, limit=10)
            )
            memory_context = build_memory_context(history_resp.turns)
            context_turns = [t.content for t in history_resp.turns[-6:]]
        except Exception as e:
            log.warning("get_history_failed", error=str(e))
            memory_context = "No previous conversation."
            context_turns = []

        # Store user turn
        try:
            memory_stub.AddConversationTurn(memory_pb2.AddTurnRequest(
                session_id=session_id,
                role="user",
                content=query,
            ))
        except Exception as e:
            log.warning("store_turn_failed", error=str(e))

        # ── Step 0.5: Input Guardrails ─────────────────────────────
        if Config.GUARDRAILS_ENABLED:
            is_safe, sanitized_query, guardrail_issues = self._check_input_guardrails(query)
            if not is_safe:
                yield agent_pb2.AgentEvent(
                    type="guardrail_blocked",
                    payload=json.dumps({
                        "reason": guardrail_issues[0].get("detail", "Input blocked by safety filter"),
                        "type": guardrail_issues[0].get("type", "unknown"),
                    }),
                )
                return
            # Log PII warnings but continue processing
            pii_warnings = [i for i in guardrail_issues if i.get("type") == "pii_warning"]
            if pii_warnings:
                log.warning("pii_detected_in_input", pii_types=pii_warnings[0].get("pii_types", []))
            query = sanitized_query  # use sanitized version going forward

        # ── Step 1: Intent Classification ─────────────────────────
        intent_result = self._classify_intent(query, memory_context)
        intent = intent_result.get("intent", "general_question")
        confidence = intent_result.get("confidence", 0.5)

        log.info("intent_classified", intent=intent, confidence=confidence)

        # ── Step 2: Check if clarification needed ─────────────────
        if intent_result.get("needs_clarification") and confidence < Config.INTENT_CONFIDENCE_THRESHOLD:
            clarification_payload = json.dumps({
                "message": intent_result.get("clarification_question", "Could you provide more details?"),
                "options": self._build_clarification_options(intent_result),
                "allow_freetext": True,
            })
            yield agent_pb2.AgentEvent(type="clarification", payload=clarification_payload)
            return

        # ── Step 2.5: Planning Layer ──────────────────────────────
        plan = None
        tool_descriptions = None
        available_tools = None
        tool_schemas = None

        if Config.PLANNING_ENABLED and intent not in ("general_question", "out_of_scope"):
            try:
                # Fetch tool descriptions early for planning
                tool_descriptions, available_tools, tool_schemas = self._get_tool_descriptions()
                available_tools_list = ", ".join(available_tools) if available_tools else "none"

                plan = self._generate_plan(query, intent, available_tools_list, memory_context)
                yield agent_pb2.AgentEvent(
                    type="agent_planning",
                    payload=json.dumps({
                        "steps": plan.get("plan_steps", []),
                        "multi_agent": plan.get("needs_multi_agent", False),
                    }),
                )
            except Exception as e:
                log.warning("planning_failed", error=str(e))

        # ── Step 3: Handle simple intents without tools ───────────
        if intent in ("general_question", "out_of_scope"):
            yield from self._handle_simple_intent(
                session_id, customer_id, query, intent, memory_context, memory_stub
            )
            return

        # ── Step 4: Execute ───────────────────────────────────────
        if (Config.MULTI_AGENT_ENABLED and plan
                and plan.get("needs_multi_agent")
                and plan.get("specialist_agents")):
            # Ensure tool info is available
            if tool_descriptions is None:
                try:
                    tool_descriptions, available_tools, tool_schemas = self._get_tool_descriptions()
                except Exception as e:
                    log.error("get_tools_failed", error=str(e))
                    yield agent_pb2.AgentEvent(
                        type="error",
                        payload=json.dumps({"message": "Tool service unavailable", "code": "TOOL_SERVICE_DOWN"}),
                    )
                    return

            yield from self._run_multi_agent_loop(
                session_id, customer_id, query, intent,
                memory_context, memory_stub, start_time, plan,
                tool_descriptions, available_tools, tool_schemas,
            )
        else:
            yield from self._run_react_loop(
                session_id, customer_id, query, intent,
                memory_context, memory_stub, start_time,
                plan=plan,
                tool_descriptions=tool_descriptions,
                available_tools=available_tools,
                tool_schemas=tool_schemas,
            )

    def _build_clarification_options(self, intent_result):
        """Build clarification options based on possible intents."""
        options = [
            {"label": "I'm looking for product recommendations", "value": "product_inquiry"},
            {"label": "I want to compare prices", "value": "price_check"},
            {"label": "I have a warranty question", "value": "warranty_question"},
            {"label": "I want to compare products", "value": "comparison"},
        ]
        return options

    def _handle_simple_intent(self, session_id, customer_id, query, intent, memory_context, memory_stub):
        """Handle intents that don't require tool use."""
        if intent == "out_of_scope":
            answer = "I'm Piper, your product support assistant. I can help you find products, compare prices, and check warranties. Is there something product-related I can help you with?"
        else:
            try:
                llm_stub = get_llm_stub()
                response = llm_stub.GenerateAnswer(llm_pb2.LLMRequest(
                    prompt=f"Conversation context:\n{memory_context}\n\nUser: {query}",
                    system_prompt="You are Piper, a friendly customer support assistant for a product catalog. Respond helpfully and concisely.",
                    temperature=0.5,
                    max_tokens=512,
                ))
                answer = response.completion or "I'm sorry, I couldn't generate a response right now. Please try again."
            except Exception as e:
                log.error("simple_intent_llm_failed", error=str(e), intent=intent)
                answer = "I'm sorry, I'm experiencing a temporary issue. Please try again in a moment."

        # Output guardrails
        if Config.GUARDRAILS_ENABLED:
            sanitized_text, was_modified, redactions = self._check_output_guardrails(answer)
            if was_modified:
                answer = sanitized_text
                log.warning("output_pii_redacted", redactions=redactions)
                yield agent_pb2.AgentEvent(
                    type="guardrail_sanitized",
                    payload=json.dumps({"redacted_types": redactions}),
                )

        # Stream tokens
        for word in answer.split(" "):
            yield agent_pb2.AgentEvent(type="token", payload=word + " ")

        # Store assistant turn
        try:
            memory_stub.AddConversationTurn(memory_pb2.AddTurnRequest(
                session_id=session_id,
                role="assistant",
                content=answer,
                intent=intent,
                confidence=1.0,
            ))
        except Exception as e:
            log.warning("store_assistant_turn_failed", error=str(e))

        # Get follow-up recommendations
        recommendations = []
        try:
            rec_stub = get_rec_stub()
            rec_response = rec_stub.GetFollowUpRecommendations(
                rec_pb2.FollowUpRecommendationRequest(
                    session_id=session_id,
                    last_query=query,
                    last_response=answer[:500],
                    intent=intent,
                    customer_id=customer_id,
                )
            )
            recommendations = list(rec_response.suggestions)
        except Exception as e:
            log.warning("get_recommendations_failed", error=str(e))

        # Response complete
        complete_payload = json.dumps({
            "response": {
                "text": answer,
                "confidence": 0.9 if intent == "general_question" else 0.5,
                "sources": [],
                "reasoning_steps": 0,
                "tools_used": [],
            },
            "recommendations": recommendations,
        })
        yield agent_pb2.AgentEvent(type="response_complete", payload=complete_payload)

    def _run_react_loop(self, session_id, customer_id, query, intent, memory_context,
                         memory_stub, start_time, plan=None, tool_descriptions=None,
                         available_tools=None, tool_schemas=None):
        """Execute the ReACT reasoning loop with tool validation, reflection, and reflexion."""
        # Get tool descriptions and schemas (skip if already passed in from planning)
        if tool_descriptions is None or available_tools is None or tool_schemas is None:
            try:
                tool_descriptions, available_tools, tool_schemas = self._get_tool_descriptions()
            except Exception as e:
                log.error("get_tools_failed", error=str(e))
                yield agent_pb2.AgentEvent(
                    type="error",
                    payload=json.dumps({"message": "Tool service unavailable", "code": "TOOL_SERVICE_DOWN"}),
                )
                return

        # ── Reflexion: retrieve past insights ─────────────────────
        reflexion_context = self._get_reflexion_insights(customer_id, intent, query)

        # Inject plan context if available
        if plan and plan.get("plan_steps"):
            plan_lines = ["\nExecution plan (follow these steps):"]
            for i, step in enumerate(plan["plan_steps"], 1):
                tool_hint = f" (use {step['suggested_tool']})" if step.get("suggested_tool") else ""
                plan_lines.append(f"  {i}. {step['goal']}{tool_hint}")
            plan_lines.append("")
            reflexion_context += "\n".join(plan_lines)

        steps = []
        tools_used = []
        final_answer = None

        for iteration in range(1, Config.REACT_MAX_ITERATIONS + 1):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > Config.REACT_TIMEOUT_SECONDS:
                log.warning("react_timeout", iteration=iteration, elapsed=elapsed)
                final_answer = "I'm sorry, I'm taking too long to process your request. Please try a simpler question."
                break

            # Build history and execute step
            react_history = build_react_history(steps)
            llm_output = self._execute_react_step(
                query, memory_context, tool_descriptions, react_history,
                reflexion_context=reflexion_context,
            )

            thought, action, action_input, answer = parse_react_output(llm_output)

            log.info(
                "react_iteration",
                iteration=iteration,
                thought=thought[:100],
                action=action,
                has_answer=answer is not None,
            )

            # Send thinking indicator
            yield agent_pb2.AgentEvent(
                type="agent_thinking",
                payload=json.dumps({"step": thought[:80], "iteration": iteration}),
            )

            if answer:
                # Agent is ready to answer
                final_answer = answer
                steps.append({
                    "iteration": iteration,
                    "thought": thought,
                })
                break

            if action and action in available_tools:
                # ── Tool Parameter Validation ─────────────────────
                if Config.TOOL_VALIDATION_ENABLED:
                    schema = tool_schemas.get(action, {})
                    is_valid, validated_params, error_msg = self._validate_tool_params(
                        action, action_input or "{}", schema
                    )
                    if not is_valid:
                        log.warning("tool_param_validation_failed",
                                    tool=action, error=error_msg)
                        yield agent_pb2.AgentEvent(
                            type="tool_validation_error",
                            payload=json.dumps({
                                "tool": action,
                                "error": error_msg,
                                "iteration": iteration,
                            }),
                        )
                        # Let the LLM self-correct via observation
                        steps.append({
                            "iteration": iteration,
                            "thought": thought,
                            "action": action,
                            "action_input": action_input,
                            "observation": f"Parameter validation error: {error_msg}. Please fix the parameters and try again.",
                        })
                        continue

                # Execute the tool
                try:
                    tool_response = self._execute_tool(session_id, action, action_input or "{}")
                    observation = tool_response.result if tool_response.success else f"Error: {tool_response.error}"
                    tools_used.append(action)

                    # ── Tool Result Validation ────────────────────
                    if Config.TOOL_VALIDATION_ENABLED and tool_response.success:
                        _, enriched_obs = self._validate_tool_result(action, observation)
                        observation = enriched_obs

                except Exception as e:
                    observation = f"Tool execution failed: {str(e)}"
                    log.error("tool_execution_failed", tool=action, error=str(e))

                steps.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })
            elif action:
                # Unknown tool
                steps.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": f"Unknown tool: {action}. Available tools: {', '.join(available_tools)}",
                })
            else:
                # No action or answer — treat raw output as answer
                final_answer = llm_output
                break

        # If we exhausted iterations without an answer, force one
        if final_answer is None:
            observations = "\n".join(
                f"- {s.get('action', 'N/A')}: {s.get('observation', 'N/A')[:200]}"
                for s in steps if s.get("observation")
            )
            final_answer = f"Based on what I found:\n{observations}\n\nPlease let me know if you need more specific information."

        # ── Frame the response ────────────────────────────────────
        try:
            framed = self._frame_response(query, final_answer, tools_used, len(steps))
        except Exception:
            framed = {"text": final_answer, "confidence": 0.7, "sources": []}

        # ── Reflection: evaluate and optionally refine ────────────
        original_score = None
        refined_score = None
        last_evaluation = {}

        if Config.REFLECTION_ENABLED:
            framed, reflection_events, original_score, last_evaluation = self._run_reflection_loop(
                query, framed, tools_used, steps, memory_context
            )
            for event in reflection_events:
                yield event

            # Capture refined score (re-evaluate is implicit from the loop)
            refined_score = framed.get("confidence", original_score)

        # ── Output Guardrails ──────────────────────────────────────
        if Config.GUARDRAILS_ENABLED:
            sanitized_text, was_modified, redactions = self._check_output_guardrails(framed["text"])
            if was_modified:
                framed["text"] = sanitized_text
                log.warning("output_pii_redacted", redactions=redactions)
                yield agent_pb2.AgentEvent(
                    type="guardrail_sanitized",
                    payload=json.dumps({"redacted_types": redactions}),
                )

        # Stream the final text as tokens
        for word in framed["text"].split(" "):
            yield agent_pb2.AgentEvent(type="token", payload=word + " ")

        # Store assistant turn
        try:
            memory_stub.AddConversationTurn(memory_pb2.AddTurnRequest(
                session_id=session_id,
                role="assistant",
                content=framed["text"],
                intent=intent,
                confidence=framed.get("confidence", 0.7),
                tool_calls=json.dumps([{"tool": t} for t in tools_used]),
            ))
        except Exception as e:
            log.warning("store_assistant_turn_failed", error=str(e))

        # ── Reflexion: store insight if quality was poor ──────────
        if Config.REFLEXION_ENABLED and original_score is not None:
            reflexion_events = self._maybe_store_reflexion_insight(
                session_id, customer_id, query, intent, tools_used,
                last_evaluation, original_score, refined_score, framed["text"],
            )
            for event in reflexion_events:
                yield event

        # ── Evaluation: store structured record ────────────────────
        if Config.EVALUATION_STORAGE_ENABLED:
            try:
                elapsed = time.time() - start_time
                self._store_evaluation_record(
                    session_id, customer_id, query, intent,
                    framed["text"], tools_used, steps,
                    framed.get("confidence", 0.7),
                    original_score, elapsed,
                )
            except Exception as e:
                log.warning("evaluation_storage_failed", error=str(e))

        # Get follow-up recommendations
        recommendations = []
        try:
            rec_stub = get_rec_stub()
            rec_response = rec_stub.GetFollowUpRecommendations(
                rec_pb2.FollowUpRecommendationRequest(
                    session_id=session_id,
                    last_query=query,
                    last_response=framed["text"][:500],
                    intent=intent,
                    customer_id=customer_id,
                )
            )
            recommendations = list(rec_response.suggestions)
        except Exception as e:
            log.warning("get_recommendations_failed", error=str(e))

        # Send response_complete event
        complete_payload = json.dumps({
            "response": {
                "text": framed["text"],
                "confidence": framed.get("confidence", 0.7),
                "sources": framed.get("sources", []),
                "reasoning_steps": len(steps),
                "tools_used": tools_used,
            },
            "recommendations": recommendations,
        })
        yield agent_pb2.AgentEvent(type="response_complete", payload=complete_payload)

    def SubmitClarification(self, request, context):
        """Handle a clarification response by re-processing with enriched context."""
        session_id = request.session_id
        selected = request.selected_option
        freetext = request.freetext

        log.info("clarification_received", session_id=session_id, selected=selected)

        memory_stub = get_memory_stub()

        # Get the last user query from history
        try:
            history_resp = memory_stub.GetConversationHistory(
                memory_pb2.GetHistoryRequest(session_id=session_id, limit=5)
            )
            # Find the last user message
            last_query = ""
            for turn in reversed(list(history_resp.turns)):
                if turn.role == "user":
                    last_query = turn.content
                    break
        except Exception:
            last_query = freetext or selected

        # Build enriched query
        if freetext:
            enriched_query = f"{last_query} — {freetext}"
        else:
            intent_map = {
                "product_inquiry": "I'm looking for product information",
                "price_check": "I want to know about prices",
                "warranty_question": "I have a warranty question",
                "comparison": "I want to compare products",
            }
            enriched_query = f"{last_query} — {intent_map.get(selected, selected)}"

        # Get session info to extract customer_id
        try:
            session_resp = memory_stub.GetSession(
                memory_pb2.GetSessionRequest(session_id=session_id)
            )
            customer_id = session_resp.customer_id
        except Exception:
            customer_id = ""

        # Re-process with the enriched query
        enriched_request = type("Request", (), {
            "session_id": session_id,
            "customer_id": customer_id,
            "query": enriched_query,
        })()

        yield from self.ProcessQuery(enriched_request, context)


# ── Server Startup ────────────────────────────────────────────────

def serve():
    with open(Config.TLS_SERVER_CERT, "rb") as f:
        server_cert = f.read()
    with open(Config.TLS_SERVER_KEY, "rb") as f:
        server_key = f.read()
    with open(Config.TLS_CA_CERT, "rb") as f:
        ca_cert = f.read()
    credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=False,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentServiceServicer(), server)
    server.add_secure_port("[::]:50054", credentials)
    server.start()
    log.info("server_started", port=50054, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
