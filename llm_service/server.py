"""LLM Service — Multi-purpose prompt management, intent classification, structured output."""

import json
import os
import re
import sys
import grpc
from concurrent import futures
from dotenv import load_dotenv
from anthropic import Anthropic

sys.path.append("..")
load_dotenv()

import protos.llm_service_pb2 as pb2
import protos.llm_service_pb2_grpc as pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging

log = setup_logging("llm_service")

client = Anthropic(api_key=Config.ANTHROPIC_API_KEY, timeout=Config.ANTHROPIC_API_TIMEOUT)

# ── Default Prompts ───────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = "You are Piper, a helpful customer support assistant for a product catalog."

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a customer support chatbot.
Classify the user's query into one of these intents:
- product_inquiry: Questions about specific products or finding products
- price_check: Questions about prices, budgets, or cost comparisons
- comparison: Requests to compare two or more products
- warranty_question: Questions about warranty duration or coverage
- general_question: Greetings, thanks, general help requests
- follow_up: References to previous conversation context
- out_of_scope: Questions not related to products or support

Respond with valid JSON only, no markdown code fences."""


# ── Helpers ───────────────────────────────────────────────────────

def build_messages(prompt: str, memory: list, system_prompt: str = None):
    """Construct the conversation history for the Claude API.

    Returns (system_str, messages_list).
    Claude uses a separate `system` parameter, so the system prompt is
    extracted rather than being the first message.
    """
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = []

    for i, turn in enumerate(memory):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})

    messages.append({"role": "user", "content": prompt})

    # Claude requires alternating user/assistant roles starting with "user".
    # If odd-length memory produces consecutive same-role messages, merge them.
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    return system, merged


# ── gRPC Service ──────────────────────────────────────────────────

class LLMServiceServicer(pb2_grpc.LLMServiceServicer):

    def GenerateAnswer(self, request, context):
        """General-purpose LLM completion with optional system prompt override."""
        prompt = request.prompt
        memory = list(request.memory)
        system_prompt = request.system_prompt or None
        temperature = request.temperature if request.temperature > 0 else Config.DEFAULT_TEMPERATURE
        max_tokens = request.max_tokens if request.max_tokens > 0 else Config.DEFAULT_MAX_TOKENS

        system, messages = build_messages(prompt, memory, system_prompt)

        log.info(
            "generate_answer",
            prompt_length=len(prompt),
            memory_turns=len(memory),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = client.messages.create(
                model=Config.LLM_MODEL,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response.content or not hasattr(response.content[0], "text"):
                log.error("generate_empty_response", stop_reason=getattr(response, "stop_reason", "unknown"))
                return pb2.LLMResponse(completion="")

            full_response = response.content[0].text
            log.info("generate_complete", response_length=len(full_response))

            return pb2.LLMResponse(completion=full_response)

        except Exception as e:
            log.error("generate_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.LLMResponse(completion="")

    def ClassifyIntent(self, request, context):
        """Dedicated intent classification endpoint."""
        query = request.query
        conversation_context = list(request.conversation_context)

        # Build context string
        context_str = ""
        if conversation_context:
            context_str = "\n".join(
                f"{'User' if i % 2 == 0 else 'Assistant'}: {turn}"
                for i, turn in enumerate(conversation_context)
            )

        prompt = (
            f"Conversation context:\n{context_str}\n\n"
            f"User query: {query}\n\n"
            "Classify the intent. Respond with valid JSON:\n"
            '{"intent": "...", "confidence": 0.0-1.0, "entities": [...], '
            '"needs_clarification": true/false, "clarification_question": "..."}'
        )

        log.info("classify_intent", query=query[:100])

        raw = ""
        try:
            response = client.messages.create(
                model=Config.LLM_MODEL,
                system=INTENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=Config.INTENT_TEMPERATURE,
                max_tokens=Config.INTENT_MAX_TOKENS,
            )

            if not response.content or not hasattr(response.content[0], "text"):
                log.error("classify_empty_response")
                return pb2.IntentResponse(intent="general_question", confidence=0.5)

            raw = response.content[0].text.strip()
            # Strip markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)

            return pb2.IntentResponse(
                intent=result.get("intent", "general_question"),
                confidence=result.get("confidence", 0.5),
                entities=result.get("entities", []),
                needs_clarification=result.get("needs_clarification", False),
                clarification_question=result.get("clarification_question", ""),
            )

        except json.JSONDecodeError:
            log.warning("intent_parse_failed", raw=raw[:200])
            return pb2.IntentResponse(
                intent="general_question",
                confidence=0.5,
            )
        except Exception as e:
            log.error("classify_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.IntentResponse(
                intent="general_question",
                confidence=0.5,
            )

    def GenerateStructured(self, request, context):
        """Generate structured JSON output from a prompt."""
        prompt = request.prompt
        system_prompt = request.system_prompt or "Respond with valid JSON only."
        temperature = request.temperature if request.temperature > 0 else 0.2
        max_tokens = request.max_tokens if request.max_tokens > 0 else 512

        log.info("generate_structured", prompt_length=len(prompt))

        raw = ""
        try:
            response = client.messages.create(
                model=Config.LLM_MODEL,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response.content or not hasattr(response.content[0], "text"):
                log.error("structured_empty_response")
                return pb2.StructuredResponse(json_output="{}")

            raw = response.content[0].text.strip()
            # Strip markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            # Validate it's valid JSON
            json.loads(raw)

            return pb2.StructuredResponse(json_output=raw)

        except json.JSONDecodeError:
            log.warning("structured_parse_failed", raw=raw[:200])
            return pb2.StructuredResponse(json_output="{}")
        except Exception as e:
            log.error("structured_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.StructuredResponse(json_output="{}")


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
    pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceServicer(), server)
    server.add_secure_port("[::]:50053", credentials)
    server.start()
    log.info("server_started", port=50053, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
