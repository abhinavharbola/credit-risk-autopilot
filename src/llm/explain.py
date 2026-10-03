import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = (
    "You are explaining one automated credit-risk-governance decision to a "
    "reviewer. You are given an event_type (gate_evaluation, promotion, "
    "rollback, rollback_check, drift_check, label_release, clock_advance, or "
    "alias_reconciled) and its JSON payload from an audit log. Write 2-3 "
    "plain-language sentences: what happened and why, grounded only in the "
    "numbers already present in the payload. No preamble, no restating the "
    "raw JSON, no speculation beyond what the payload shows."
)


def explain_event(event_type: str, payload: dict[str, Any]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "LLM explanation unavailable: GROQ_API_KEY is not set."

    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"event_type: {event_type}\n"
                        f"payload: {json.dumps(payload, default=str)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return "LLM explanation unavailable: the model returned no text."
        return content
    except Exception as e:
        return f"LLM explanation unavailable: {e}"
