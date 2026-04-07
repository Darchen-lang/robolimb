"""
Configuration management for RoboLimb speech listener.
Loads from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000
    blocksize: int = 8000
    channels: int = 1
    dtype: str = "int16"
    max_queue_size: int = 20

    @classmethod
    def from_env(cls) -> "AudioConfig":
        return cls(
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "20")),
        )


@dataclass
class VoskConfig:
    """Vosk speech recognition settings."""
    model_path: str = "vosk-model-small-en-in-0.4"
    max_recording_duration: int = 15

    @classmethod
    def from_env(cls) -> "VoskConfig":
        return cls(
            model_path=os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-in-0.4"),
            max_recording_duration=int(os.getenv("MAX_RECORDING_DURATION", "15")),
        )


@dataclass
class OpenAIConfig:
    """OpenAI API settings."""
    api_key: Optional[str] = None
    model: str = "gpt-4-mini"
    timeout_seconds: float = 10.0
    max_retries: int = 3
    temperature: float = 0.3

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4-mini"),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT", "10.0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        )


@dataclass
class Config:
    """Global configuration."""
    audio: AudioConfig
    vosk: VoskConfig
    openai: OpenAIConfig

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            audio=AudioConfig.from_env(),
            vosk=VoskConfig.from_env(),
            openai=OpenAIConfig.from_env(),
        )

    def validate(self) -> bool:
        """Validate critical configuration."""
        if not self.openai.api_key:
            print("❌ Error: OPENAI_API_KEY not set in .env file")
            return False
        return True
