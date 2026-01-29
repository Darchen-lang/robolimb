import os
import queue
import sounddevice as sd
import vosk
import json
import keyboard
import sys
from contextlib import suppress
from time import sleep
from sys import platform
import time
from parser import parse_text_with_openai


MODEL_PATH = "vosk-model-small-en-in-0.4" 
SAMPLE_RATE = 16000 

# Bounded queue to avoid unbounded memory growth on audio backlog.
# Maxsize can be tuned based on acceptable latency and memory usage.
audio_queue = queue.Queue(maxsize=20)

def audio_callback(indata, frames, time_info, status):  
    """This function puts raw audio chunks from the mic into the queue"""
    if status:
        print(status, file=sys.stderr)

    chunk = bytes(indata)
    try:
        # Non-blocking put: if the queue is full, niche handle kr lenge.
        audio_queue.put_nowait(chunk)
    except queue.Full:
        # Drop the oldest item to make room for the latest audio chunk.
        with suppress(queue.Empty):
            audio_queue.get_nowait()
        # Best-effort enqueue of the latest chunk.
        with suppress(queue.Full):
            audio_queue.put_nowait(chunk)

def _drain_audio_queue(q):
    """Drain all items from the queue to clear old audio data"""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break

def _handle_recognition_result(result_json: str, transcription: list) -> None:
    """Handle the final Vosk result: print, store, and route intents."""
    result_dict = json.loads(result_json)

    if not (text := result_dict.get("text", "")):
        print(" (No speech detected)               ")
        return

    normalized_text = text.strip()
    print(f"Captured: {normalized_text}            ")
    transcription.append(normalized_text)

    # OpenAI se parsing
    parsed = parse_text_with_openai(normalized_text)
    print(f"Parsed: {parsed}")
    intent = parsed.get("intent")

    # Simple intent routing – extend as needed
    if intent == "open":
        print("Command Detected: OPEN GRIPPER (via OpenAI)")
        # TODO: call your robot's 'open' action here
    elif intent == "close":
        print("Command Detected: CLOSE GRIPPER (via OpenAI)")
        # TODO: call your robot's 'close' action here
    elif intent == "error":
        print(f"Parser error: {parsed.get('error')} - {parsed.get('message')}")
    else:
        print(f"No known command for intent: {intent}")


def _record_until_space_released(model: vosk.Model, max_duration_seconds: int) -> str:
    """
    Create a recognizer and feed it audio until:
    - space is released, or
    - max_duration_seconds is reached.

    Returns the final Vosk result JSON string.
    """
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

    # Clear any old audio from the queue
    _drain_audio_queue(audio_queue)

    start_time = time.time()

    while keyboard.is_pressed("space"):
        if time.time() - start_time > max_duration_seconds:
            print(f"\n Maximum recording time ({max_duration_seconds}s) reached!")
            break

        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        recognizer.AcceptWaveform(data)

    return recognizer.FinalResult()


def main() -> None:
    """Main loop: listen on mic, recognize speech, and route commands."""
    # Initialize Vosk model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model folder '{MODEL_PATH}' not found!")
        return

    model = vosk.Model(MODEL_PATH)
    transcription: list[str] = []

    print("--- RoboLimb Active ---")
    print("Ready. HOLD 'SPACE' to speak...")

    max_duration_seconds = 15  # maximum recording duration

    # Open the Microphone Stream using sounddevice
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            try:
                if keyboard.is_pressed("space"):
                    print(" Listening...", end="\r")

                    result_json = _record_until_space_released(
                        model=model,
                        max_duration_seconds=max_duration_seconds,
                    )
                    _handle_recognition_result(result_json, transcription)
                else:
                    sleep(0.05)

            except KeyboardInterrupt:
                break

    print("\n\nFinal Transcription History:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()