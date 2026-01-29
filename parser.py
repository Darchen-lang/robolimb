import os
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None


def parse_text_with_openai(text: str) -> Dict[str, Any]:
    """
    Send recognized text to OpenAI for parsing / intent classification / etc.
    Returns the parsed result as a Python dict.
    """
    if not text.strip():
        return {"intent": "empty", "raw": text, "reason": "no text provided"}

    if client is None:
        return {
            "intent": "error",
            "raw": text,
            "error": "no_api_key",
            "message": "OPENAI_API_KEY not configured",
        }

    # Example: ask OpenAI to extract intent & arguments
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a command parser for a robotic limb. "
                        "Given user speech, extract a structured command."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User said: {text}\nReturn JSON with keys: intent (string), "
                        "arguments (object), confidence (0-1)."
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return {
            "intent": "error",
            "raw": text,
            "error": "openai_request_failed",
            "message": str(exc),
        }

    # The SDK returns content pieces; pick the first text item
    content = response.output[0].content[0].text
    # content is already JSON text because we requested json_object
    import json
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"intent": "unknown", "raw": text, "error": "invalid_json_from_model"}

    return parsed
