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

from nlp_parser import parse_text_with_openai

# Add parent directory to path to import robot_coordinator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_coordinator import coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use absolute path for model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", os.path.join(SCRIPT_DIR, "vosk-model-small-en-in-0.4"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "20"))
MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", "15"))

# Audio queue
audio_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# Parser queue: recognized text waits here to be parsed by a worker thread
parser_queue: queue.Queue = queue.Queue(maxsize=50)

# Recording control queue: press Enter to start, Enter again to stop
recording_control_queue: queue.Queue = queue.Queue(maxsize=10)

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
    """
    Route parsed intents to robot actions.
    Communicates with the robot coordinator to queue commands.
    """
    intent_lower = intent.lower().strip()
    arguments = arguments or {}
    
    if intent_lower in ("pick", "pick_object", "grab", "pick up"):
        # Extract target object from arguments
        target = arguments.get("target_object") or arguments.get("object")
        
        if target:
            target_str = str(target).lower().strip()
            logger.info(f"PICK command for: {target_str}")
            print(f"🎯 Picking up: {target_str}")
            
            confidence = float(arguments.get("confidence", 0.75))
            # Queue pick command to coordinator (will be handled by vision module)
            coordinator.queue_pick_command(target_str, confidence)
        else:
            logger.warning("Pick command received but no target object specified")
            print("⚠️ Pick command received but no target object specified")
    
    elif intent_lower == "open":
        logger.info("Command: OPEN GRIPPER")
        print("→ Command: OPEN GRIPPER")
        coordinator.queue_command("open_gripper")
    
    elif intent_lower == "close":
        logger.info("Command: CLOSE GRIPPER")
        print("→ Command: CLOSE GRIPPER")
        coordinator.queue_command("close_gripper")
    
    elif intent_lower == "home":
        logger.info("Command: RETURN HOME")
        print("🏠 Returning home...")
        coordinator.queue_command("home")
    
    elif intent_lower == "stop":
        logger.info("Command: STOP")
        print("⏹️  Stopping operations...")
        coordinator.queue_command("stop")
    
    elif intent_lower in ("unknown", "empty"):
        logger.debug("No recognized intent")
    
    else:
        logger.info(f"Unhandled intent: {intent}")
        print(f"❓ Unhandled intent: {intent} (args: {arguments})")




def control_input_worker() -> None:
    """
    Terminal control worker:
    - Press Enter to start recording
    - Press Enter again to stop recording
    - Type 'q' + Enter to quit
    """
    while not stop_event.is_set():
        try:
            user_input = input().strip().lower()
        except EOFError:
            logger.info("Input stream closed; stopping listener")
            stop_event.set()
            break
        except Exception as e:
            logger.warning(f"Input worker error: {e}")
            continue

        if user_input in ("q", "quit", "exit"):
            stop_event.set()
            break

        # Empty line means Enter key press for recording control.
        if user_input == "":
            with suppress(queue.Full):
                recording_control_queue.put_nowait("toggle")


def record_until_toggled_or_timeout(
    model: vosk.Model,
    max_duration_seconds: int,
) -> str:
    """
    Create a recognizer and feed it audio until:
    - Enter is pressed again (toggle stop), or
    - max_duration_seconds is reached.

    Returns the final Vosk result JSON string.
    """
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    drain_audio_queue(audio_queue)
    drain_audio_queue(recording_control_queue)
    
    start_time = time.time()

    while not stop_event.is_set():
        # A second Enter stops the current recording.
        try:
            recording_control_queue.get_nowait()
            logger.debug("Recording stopped by Enter key")
            break
        except queue.Empty:
            pass

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
    print("Press Enter to start recording (max 15s)")
    print("Press Enter again to stop recording")
    print("Type 'q' then Enter to quit speech listener")
    print("Press Ctrl+C to exit")
    print("=" * 50 + "\n")

    try:
        # Start terminal input worker for Enter-based recording control.
        control_thread = threading.Thread(
            target=control_input_worker,
            daemon=True,
            name="ControlInput",
        )
        control_thread.start()

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
            print("Press Enter to start your first recording...\n")

            while not stop_event.is_set():
                try:
                    recording_control_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                is_recording = True
                recording_start_time = time.time()
                print("\n⏺️  Listening... (press Enter to stop)")
                logger.debug("Recording started")

                result_json = record_until_toggled_or_timeout(
                    model=model,
                    max_duration_seconds=MAX_RECORDING_DURATION,
                )
                handle_recognition_result(result_json, transcription)
                is_recording = False
                print("\nPress Enter to record again...\n")

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