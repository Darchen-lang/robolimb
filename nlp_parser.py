import os
import json
import logging
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None


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


def parse_text_with_openai(
    text: str,
    max_retries: int = 3,
    timeout_seconds: float = 10.0,
) -> ParsedCommand:
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
        return ParsedCommand(
            intent="error",
            raw_text=text,
            error="no_api_key",
            message="OPENAI_API_KEY not configured",
        )

    # Sanitize text
    sanitized_text = text.strip()[:1000]  # Limit to 1000 chars
    system_prompt = (
        "You are a command parser for a robotic limb. "
        "Given user speech, extract a structured command. "
        "Return ONLY valid JSON with keys: intent (string), arguments (object), confidence (0-1)."
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
            return ParsedCommand(
                intent="error",
                raw_text=sanitized_text,
                error="rate_limit",
                message="API rate limit exceeded",
            )

        except (APIConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return ParsedCommand(
                intent="error",
                raw_text=sanitized_text,
                error="connection_error",
                message=f"Failed to reach OpenAI API: {str(e)}",
            )

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return ParsedCommand(
                intent="error",
                raw_text=sanitized_text,
                error="openai_error",
                message=str(e),
            )

        except Exception as e:
            logger.error(f"Unexpected error during parsing: {e}", exc_info=True)
            return ParsedCommand(
                intent="error",
                raw_text=sanitized_text,
                error="unexpected",
                message=str(e),
            )

    # Should not reach here
    return ParsedCommand(
        intent="error",
        raw_text=sanitized_text,
        error="max_retries",
        message=f"Exceeded max retries ({max_retries})",
    )
