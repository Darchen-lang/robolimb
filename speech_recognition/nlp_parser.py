import os
import json
import logging
import time
import re
from typing import Any, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DOTENV = os.path.join(WORKSPACE_ROOT, ".env")

# Load root .env explicitly because runtime code changes cwd.
load_dotenv(dotenv_path=ROOT_DOTENV)

logger = logging.getLogger(__name__)

_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
_openai_init_error: Optional[str] = None
client = None
if _api_key:
    try:
        client = OpenAI(api_key=_api_key)
    except Exception as e:
        _openai_init_error = str(e)
        logger.warning(f"OpenAI client initialization failed; using local parser fallback: {_openai_init_error}")


@dataclass
class ParsedCommand:
    """Validated response from OpenAI parser."""
    intent: str
    arguments: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    raw_text: str = ""
    error: Optional[str] = None
    message: Optional[str] = None

    def is_error(self) -> bool:
        return self.error is not None


def parse_text_locally(text: str) -> ParsedCommand:
    """Best-effort local parser used when OpenAI is unavailable."""
    raw_text = text.strip()
    normalized = raw_text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ParsedCommand(
            intent="empty",
            raw_text=raw_text,
            error="empty_input",
            message="No text provided",
        )

    if "stop" in normalized:
        return ParsedCommand(intent="stop", arguments={}, confidence=0.9, raw_text=raw_text)

    if any(token in normalized for token in ("home", "go home", "return home")):
        return ParsedCommand(intent="home", arguments={}, confidence=0.9, raw_text=raw_text)

    if "open" in normalized:
        return ParsedCommand(intent="open", arguments={}, confidence=0.85, raw_text=raw_text)

    if any(token in normalized for token in ("close", "shut")):
        return ParsedCommand(intent="close", arguments={}, confidence=0.85, raw_text=raw_text)

    pick_words = ("pick", "grab", "take", "lift")
    if any(word in normalized for word in pick_words):
        target = normalized
        for phrase in (
            "pick up the", "pick up", "pick the", "pick",
            "grab the", "grab", "take the", "take", "lift the", "lift",
            "please", "can you", "could you",
        ):
            target = target.replace(phrase, " ")
        target = re.sub(r"\b(it|this|that|object)\b", " ", target)
        target = re.sub(r"\s+", " ", target).strip()
        arguments: Dict[str, Any] = {}
        if target:
            arguments["target_object"] = target
        return ParsedCommand(
            intent="pick",
            arguments=arguments,
            confidence=0.8 if target else 0.65,
            raw_text=raw_text,
        )

    return ParsedCommand(intent="unknown", arguments={}, confidence=0.3, raw_text=raw_text)


def parse_text_with_openai(
    text: str,
    max_retries: int = 3,
    timeout_seconds: float = 10.0,
) -> ParsedCommand:  # sourcery skip: low-code-quality
    """
    Send recognized text to OpenAI for parsing / intent classification.
    
    Args:
        text: The recognized speech text to parse.
        max_retries: Number of retry attempts on transient errors.
        timeout_seconds: Timeout for the API request.
    
    Returns:
        ParsedCommand with intent, arguments, and optional error info.
    """
    if not text.strip():
        return ParsedCommand(
            intent="empty",
            raw_text=text,
            error="empty_input",
            message="No text provided",
        )

    if client is None:
        local_result = parse_text_locally(text)
        if _api_key and _openai_init_error:
            logger.info(f"OpenAI unavailable ({_openai_init_error}); using local parser fallback")
        else:
            logger.info("OPENAI_API_KEY not configured; using local parser fallback")
        return local_result

    # Sanitize text
    sanitized_text = text.strip()[:1000]  # Limit to 1000 chars
    system_prompt = (
        "You are a command parser for a robotic arm that picks up objects. "
        "Given user speech, extract a structured command. "
        "Return ONLY valid JSON with keys: intent (string), arguments (object), confidence (0-1). "
        "\n"
        "Intents can be: 'pick', 'open', 'close', 'home', 'stop', or 'unknown'. "
        "For 'pick' intent, include target_object with the object name in arguments. "
        "For example: if user says 'pick up the cup', return intent='pick' with arguments={'target_object': 'cup'}. "
        "Examples: 'grab the ball' → intent='pick', arguments={'target_object': 'ball'}, confidence=0.95. "
        "'open gripper' → intent='open', arguments={}, confidence=0.95."
    )
    user_prompt = (
        f"User said: {sanitized_text}\n"
        "Return JSON with keys: intent (string), arguments (object), confidence (0-1)."
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=timeout_seconds,
            )
            
            # Validate response structure
            if not response or not response.choices or not response.choices[0].message:
                logger.warning("Invalid OpenAI response structure")
                return ParsedCommand(
                    intent="error",
                    raw_text=sanitized_text,
                    error="invalid_response",
                    message="Empty or malformed response from OpenAI",
                )
            
            content = response.choices[0].message.content
            if not isinstance(content, str):
                logger.warning("Response content is not a string")
                return ParsedCommand(
                    intent="error",
                    raw_text=sanitized_text,
                    error="invalid_response",
                    message="Response is not text",
                )

            # Parse JSON
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return ParsedCommand(
                    intent="error",
                    raw_text=sanitized_text,
                    error="invalid_json",
                    message=f"Model returned invalid JSON: {str(e)}",
                )

            # Validate required fields
            intent = parsed_json.get("intent")
            if not isinstance(intent, str) or not intent.strip():
                logger.warning("Missing or invalid intent in model output")
                intent = "unknown"

            confidence = parsed_json.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            confidence = max(0.0, min(1.0, float(confidence)))

            arguments = parsed_json.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            result = ParsedCommand(
                intent=intent,
                arguments=arguments,
                confidence=confidence,
                raw_text=sanitized_text,
            )
            logger.debug(f"Parsed command: {result.intent} (confidence: {result.confidence})")
            return result

        except RateLimitError as e:
            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying...")
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                time.sleep(wait)
                continue
            logger.warning("OpenAI rate limited; falling back to local parser")
            return parse_text_locally(sanitized_text)

        except (APIConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            logger.warning("OpenAI unavailable; falling back to local parser")
            return parse_text_locally(sanitized_text)

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            logger.warning("OpenAI API error; falling back to local parser")
            return parse_text_locally(sanitized_text)

        except Exception as e:
            logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            logger.warning("Unexpected parser error; falling back to local parser")
            return parse_text_locally(sanitized_text)

    # Should not reach here
    logger.warning("Exceeded parser retries; falling back to local parser")
    return parse_text_locally(sanitized_text)
