# RoboLimb Speech Listener

A Python application that listens for speech via microphone, transcribes it using Vosk, and uses OpenAI's GPT-4 to parse voice commands for robotic limb control.

## Features

- **Real-time speech recognition** using Vosk (offline, fast)
- **Intent parsing** via OpenAI API (NLP with high accuracy)
- **Non-blocking architecture** with worker threads to avoid audio lag
- **Configurable via environment variables**
- **Comprehensive logging** for debugging and monitoring
- **Graceful error handling** with retries and timeouts
- **Clean command routing** for custom robot actions

## Architecture

```
Microphone → Audio Queue → Vosk Recognizer → Text Queue → OpenAI Parser → Intent Router → Robot Commands
     ↑                                                           ↑
     └──────────────────────────────────────────────────────────┘
                    Main Thread                   Worker Thread
```

The architecture uses:
- **Main thread**: Captures audio and performs speech recognition (audio-critical path)
- **Worker thread**: Handles OpenAI API calls asynchronously (non-blocking)

This separation prevents network latency from blocking audio capture.

## Prerequisites

### System Requirements

- Python 3.8+
- Microphone input device
- Linux, macOS, or Windows

### Python Dependencies

See `requirements.txt`:
- `python-dotenv` – Load API keys from `.env` file
- `openai` – OpenAI API client
- `sounddevice` – Audio capture library
- `vosk` – Offline speech recognition
- `pynput` – Keyboard input (cross-platform)

### Optional: Audio Library Requirements

On Linux, `sounddevice` may require:
```bash
sudo apt-get install libsndfile1
```

On macOS:
```bash
brew install libsndfile
```

## Setup

### 1. Clone or Download

```bash
cd /home/samaksh/Desktop/robo
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The command `pip install dotenv` fails; use `pip install python-dotenv` instead.

### 4. Download Vosk Model

Download the Vosk model and extract it to the project directory:

```bash
# Download (example for English-Indian model)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip
unzip vosk-model-small-en-in-0.4.zip
```

The model directory should match the path in your environment (default: `vosk-model-small-en-in-0.4`).

### 5. Configure API Key

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Getting an API key:**
1. Go to https://platform.openai.com/api-keys
2. Create a new secret key
3. Add it to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Usage

### Run the Listener

```bash
python vosk_listener.py
```

### Interactive Use

1. **Wait for the prompt:** "Hold 'SPACE' to speak"
2. **Hold SPACE** and speak your command (max 15 seconds)
3. **Release SPACE** when done speaking
4. The app will:
   - Transcribe your speech with Vosk
   - Send the text to OpenAI for intent parsing
   - Extract intent and arguments (e.g., "open", "close")
   - Print recognized intent and route to robot commands
5. **Repeat** for more commands
6. **Ctrl+C** to exit

### Example Commands

Try saying:
- "Open the gripper"
- "Close the gripper"
- "Move forward"

The OpenAI parser will recognize these as intents like `open`, `close`, `move`, etc.

## Configuration

Customize via environment variables in `.env` or export them:

```bash
export VOSK_MODEL_PATH="vosk-model-small-en-in-0.4"
export SAMPLE_RATE=16000
export MAX_QUEUE_SIZE=20
export MAX_RECORDING_DURATION=15
python vosk_listener.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `VOSK_MODEL_PATH` | `vosk-model-small-en-in-0.4` | Path to Vosk model directory |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `MAX_QUEUE_SIZE` | `20` | Max audio chunks in queue |
| `MAX_RECORDING_DURATION` | `15` | Max recording time in seconds |

## Code Structure

### `nlp_parser.py`

Handles OpenAI API communication:
- `parse_text_with_openai(text)` – Main parsing function
- Performs retries with exponential backoff
- Validates response structure
- Returns `ParsedCommand` dataclass with intent, arguments, confidence, and error info

**Key features:**
- Type hints and dataclass validation
- Proper exception handling (rate limits, connection errors)
- Configurable timeouts and retries
- Detailed logging

### `vosk_listener.py`

Main speech listening and routing:
- `audio_callback()` – Sounddevice callback for audio capture
- `drain_audio_queue()` – Clear stale audio
- `parse_worker()` – Background thread for OpenAI calls
- `record_until_space_released()` – Capture audio while SPACE held
- `handle_recognition_result()` – Process Vosk output, enqueue for parsing
- `_route_intent()` – Custom intent routing (extend for your robot)
- `main()` – Main loop and orchestration

**Threading model:**
- Main thread: Audio capture and Vosk recognition (real-time)
- Worker thread: OpenAI API calls (non-blocking)

## Logging

Logs are printed to console with timestamp, level, and module:

```
2025-01-15 10:23:45,123 - vosk_listener - INFO - Microphone stream opened
2025-01-15 10:23:47,456 - nlp_parser - INFO - Parsed command: open (confidence: 0.95)
```

Adjust log level in `vosk_listener.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## Extending with Robot Commands

To integrate with your robot, edit `_route_intent()` in `vosk_listener.py`:

```python
def _route_intent(intent: str, arguments: Optional[dict]) -> None:
    if intent_lower == "open":
        my_robot.open_gripper()  # Call your robot API
    elif intent_lower == "close":
        my_robot.close_gripper()
    # Add more intents...
```

## Troubleshooting

### "Model folder not found"
- Ensure Vosk model is downloaded and extracted in the project directory
- Or set `VOSK_MODEL_PATH` to the correct path

### "OPENAI_API_KEY not configured"
- Create `.env` file with your API key (see Setup section)
- Or export: `export OPENAI_API_KEY=sk-...`

### "No speech detected" repeatedly
- Check microphone input levels
- Ensure audio is loud enough
- Try moving closer to microphone

### Keyboard input not working (Linux)
- `pynput` may require additional permissions on Linux
- Run with: `sudo python vosk_listener.py` (not ideal, but works)
- Alternatively, modify code to use a GUI button or stdin instead of keyboard

### "Rate limit exceeded"
- Reduce frequency of commands
- App automatically retries with exponential backoff
- Consider upgrading OpenAI API plan for higher limits

### Connection timeouts
- Check internet connection
- Increase `timeout_seconds` in `nlp_parser.py`
- App automatically retries failed requests

## File Structure

```
robo/
├── nlp_parser.py              # OpenAI integration with retries & validation
├── vosk_listener.py           # Main speech listener with worker threads
├── vosk-model-small-en-in-0.4/  # Vosk model files (download)
├── requirements.txt           # Python dependencies
├── .env                       # API keys (local, not in git)
├── .env.example              # Template for .env
└── README.md                 # This file
```

## Performance Notes

- **Vosk**: Fast offline recognition (~50-200ms per phrase)
- **OpenAI API**: Network dependent (1-3 seconds typical)
- **Audio capture**: Continuous, bounded queue prevents memory overflow
- **Worker thread**: Prevents OpenAI lag from blocking audio

## Future Improvements

- [ ] Custom intent schema validation (enforce expected arguments)
- [ ] Confidence thresholding (ignore low-confidence parses)
- [ ] Multi-language support
- [ ] Better Vosk model selection (accuracy vs. speed)
- [ ] Web UI or REST API for remote control
- [ ] Audio recording and playback for debugging
- [ ] Metrics and performance monitoring

## License

(Specify your license here, e.g., MIT)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs for error messages
3. Verify dependencies: `pip list`
4. Test microphone: `python -c "import sounddevice; print(sounddevice.query_devices())"`
