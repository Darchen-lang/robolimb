import os
import queue
import threading
import logging
import time
import json
import sys
from contextlib import suppress
from typing import List, Optional

import sounddevice as sd
import vosk
from pynput import keyboard

from nlp_parser import parse_text_with_openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-in-0.4")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "20"))
MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", "15"))

# Audio queue
audio_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# Parser queue: recognized text waits here to be parsed by a worker thread
parser_queue: queue.Queue = queue.Queue(maxsize=50)

# Control: signal to stop the main loop
stop_event = threading.Event()

# Recording state
is_recording = False
recording_start_time: Optional[float] = None


def audio_callback(indata, frames, time_info, status):
    """Put raw audio chunks from the mic into the queue."""
    if status:
        logger.warning(f"Audio callback status: {status}")

    chunk = bytes(indata)
    try:
        audio_queue.put_nowait(chunk)
    except queue.Full:
        # Drop oldest to make room for latest
        with suppress(queue.Empty):
            audio_queue.get_nowait()
        with suppress(queue.Full):
            audio_queue.put_nowait(chunk)


def drain_audio_queue(q: queue.Queue) -> None:
    """Drain all items from the queue to clear old audio data."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def parse_worker(transcription: List[str]) -> None:
    """
    Worker thread: continuously consume recognized text from parser_queue,
    send to OpenAI, and print results without blocking audio capture.
    """
    logger.info("Parser worker started")
    while not stop_event.is_set():
        try:
            # Wait for text with timeout so we can check stop_event
            text = parser_queue.get(timeout=0.5)
            logger.debug(f"Parsing: {text}")
            
            parsed = parse_text_with_openai(text, max_retries=3, timeout_seconds=10.0)
            
            if parsed.is_error():
                logger.error(
                    f"Parser error: {parsed.error} - {parsed.message}"
                )
                print(f"❌ Parser error: {parsed.error} - {parsed.message}")
            else:
                logger.info(
                    f"Intent: {parsed.intent} (confidence: {parsed.confidence})"
                )
                print(f"✓ Intent: {parsed.intent} (conf: {parsed.confidence})")
                
                # Route intent
                _route_intent(parsed.intent, parsed.arguments)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Parser worker error: {e}", exc_info=True)


def _route_intent(intent: str, arguments: Optional[dict]) -> None:
    """Simple intent routing. Extend as needed."""
    intent_lower = intent.lower().strip()
    
    if intent_lower == "open":
        logger.info("Command: OPEN GRIPPER")
        print("→ Command: OPEN GRIPPER")
        # TODO: call robot's 'open' action
    elif intent_lower == "close":
        logger.info("Command: CLOSE GRIPPER")
        print("→ Command: CLOSE GRIPPER")
        # TODO: call robot's 'close' action
    elif intent_lower == "unknown" or intent_lower == "empty":
        logger.debug(f"No recognized intent")
    else:
        logger.info(f"Unhandled intent: {intent}")
        print(f"? Unhandled intent: {intent}")




def record_until_space_released(
    model: vosk.Model,
    max_duration_seconds: int,
) -> str:
    """
    Create a recognizer and feed it audio until:
    - space is released, or
    - max_duration_seconds is reached.

    Returns the final Vosk result JSON string.
    """
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    drain_audio_queue(audio_queue)
    
    start_time = time.time()

    while True:
        # Check if space is still pressed
        try:
            space_pressed = keyboard.is_pressed("space")
        except Exception as e:
            logger.warning(f"Keyboard error: {e}, assuming space released")
            space_pressed = False

        if not space_pressed:
            break

        elapsed = time.time() - start_time
        if elapsed > max_duration_seconds:
            logger.warning(
                f"Max recording time ({max_duration_seconds}s) reached"
            )
            break

        try:
            data = audio_queue.get(timeout=0.1)
            recognizer.AcceptWaveform(data)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error during recognition: {e}")
            break

    result_json = recognizer.FinalResult()
    logger.debug(f"Vosk result: {result_json}")
    return result_json


def handle_recognition_result(result_json: str, transcription: List[str]) -> None:
    """
    Handle the final Vosk result: validate, store, and queue for parsing.
    Non-blocking: just enqueues work for the parser worker.
    """
    try:
        result_dict = json.loads(result_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid Vosk result JSON: {e}")
        return

    text = result_dict.get("text", "").strip()
    if not text:
        logger.debug("No speech detected")
        print("(silence)")
        return

    logger.info(f"Captured: {text}")
    print(f"📝 Captured: {text}")
    transcription.append(text)

    # Enqueue for non-blocking parsing
    try:
        parser_queue.put_nowait(text)
    except queue.Full:
        logger.warning("Parser queue full, dropping oldest item")
        with suppress(queue.Empty):
            parser_queue.get_nowait()
        with suppress(queue.Full):
            parser_queue.put_nowait(text)


def main() -> None:
    """Main loop: listen on mic, recognize speech, and route commands."""
    global is_recording, recording_start_time

    # Validate model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model folder '{MODEL_PATH}' not found!")
        print(f"Error: Model folder '{MODEL_PATH}' not found!")
        return

    try:
        model = vosk.Model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        print(f"Error: Failed to load Vosk model: {e}")
        return

    transcription: List[str] = []

    # Start parser worker thread
    parser_thread = threading.Thread(
        target=parse_worker,
        args=(transcription,),
        daemon=False,
    )
    parser_thread.start()
    logger.info("Parser worker thread started")

    print("\n" + "=" * 50)
    print("🤖 RoboLimb Speech Listener")
    print("=" * 50)
    print("Hold 'SPACE' to speak (max 15s)")
    print("Press Ctrl+C to exit")
    print("=" * 50 + "\n")

    try:
        # Open mic stream
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            logger.info("Microphone stream opened")
            print("✓ Microphone ready\n")

            while not stop_event.is_set():
                try:
                    space_pressed = keyboard.is_pressed("space")
                except Exception as e:
                    logger.debug(f"Keyboard check failed: {e}")
                    space_pressed = False

                if space_pressed:
                    if not is_recording:
                        is_recording = True
                        recording_start_time = time.time()
                        print("\n⏺️  Listening...")
                        logger.debug("Recording started")

                    result_json = record_until_space_released(
                        model=model,
                        max_duration_seconds=MAX_RECORDING_DURATION,
                    )
                    handle_recognition_result(result_json, transcription)
                    is_recording = False

                else:
                    if is_recording:
                        is_recording = False
                    time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\n⏹️  Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        stop_event.set()
        parser_thread.join(timeout=5.0)
        logger.info("Parser thread joined")

        # Print summary
        print("\n" + "=" * 50)
        print("📋 Transcription History")
        print("=" * 50)
        if transcription:
            for i, line in enumerate(transcription, 1):
                print(f"{i}. {line}")
        else:
            print("(No speech captured)")
        print("=" * 50 + "\n")
        logger.info("RoboLimb listener stopped")


if __name__ == "__main__":
    main()