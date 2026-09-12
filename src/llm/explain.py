"""Single isolated, stateless call (Groq) to explain one logged decision in
plain language. Supporting feature only, not a pillar of the project
(section 3): nothing in the governance loop calls this or blocks on it -
it's read-only sugar for the audit_log dashboard view (dashboard/views/
audit_log.py).

Fails soft, never raises: a missing GROQ_API_KEY, a missing `groq` install,
or an API error all return a short placeholder string instead of an
exception, since a broken LLM call should never take down the audit log
view it's decorating.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = (
    "You are explaining one automated credit-risk-governance decision to a "
    "reviewer. You are given an event_type (gate_evaluation, promotion, "
    "rollback, rollback_check, drift_check, or label_release) and its JSON "
    "payload from an audit log. Write 2-3 plain-language sentences: what "
    "happened and why, grounded only in the numbers already present in the "
    "payload. No preamble, no restating the raw JSON, no speculation beyond "
    "what the payload shows."
)


def explain_event(event_type: str, payload: dict[str, Any]) -> str:
    """Returns a short plain-language explanation of one audit_log event."""
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
                    "content": f"event_type: {event_type}\npayload: {payload}",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM explanation unavailable: {e}"
