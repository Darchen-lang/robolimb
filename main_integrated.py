#!/usr/bin/env python3
"""
ROBOLIMB INTEGRATED CONTROLLER
Main entry point that launches both speech recognition and computer vision modules
in separate threads, allowing voice control of object picking.

Usage: python main_integrated.py [--speech-only] [--vision-only]
"""

import os
import sys
import threading
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(threadName)s] - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add workspace path for imports
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _setup_module_path(module_dir):
    """Setup Python path and change directory for a module."""
    os.chdir(os.path.join(WORKSPACE_ROOT, module_dir))
    sys.path.insert(0, os.path.join(WORKSPACE_ROOT, module_dir))


def _load_module(module_name, module_dir, file_name):
    """Load a module dynamically using importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(WORKSPACE_ROOT, module_dir, file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_speech_module():
    """Run speech recognition module in its own thread."""
    logger.info("Initializing speech recognition module...")
    try:
        _setup_module_path("speech_recognition")
        vosk_listener_module = _load_module("vosk_listener", "speech_recognition", "vosk_listener.py")
        logger.info("Starting speech recognition listener...")
        vosk_listener_module.main()
    except Exception as e:
        logger.error(f"Speech module error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Speech recognition module stopped")


def run_vision_module():
    """Run computer vision module in its own thread."""
    logger.info("Initializing computer vision module...")
    try:
        _setup_module_path("computer_vision")
        robolimb_module = _load_module("robolimb", "computer_vision", "robolimb.py")
        logger.info("Starting computer vision detector...")
        robolimb_module.main()
    except Exception as e:
        logger.error(f"Vision module error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Computer vision module stopped")


def _launch_module_thread(threads, target_func, thread_name, description):
    """Launch a module in a separate thread."""
    thread = threading.Thread(
        target=target_func,
        name=thread_name,
        daemon=False
    )
    thread.start()
    threads.append(thread)
    print(f"✓ {description} thread started")
    time.sleep(1)  # Give module time to initialize


def main():
    """Main entry point - launch both modules."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RoboLimb Integrated Controller with Voice Control"
    )
    parser.add_argument(
        "--speech-only",
        action="store_true",
        help="Run only speech recognition module"
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Run only computer vision module"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🤖 ROBOLIMB - INTEGRATED VOICE-CONTROLLED PICKER")
    print("=" * 70)
    print("\nSystem Architecture:")
    print("  ├─ Speech Recognition (Vosk + OpenAI)")
    print("  ├─ Computer Vision (YOLOv8)")
    print("  ├─ 3D Coordinate Estimation")
    print("  ├─ Inverse Kinematics")
    print("  └─ Arduino Servo Control")
    print("\nIntegration:")
    print("  • Speech commands are parsed into structured intents")
    print("  • Pick commands identify target objects by name")
    print("  • Vision system detects and locates objects in real-time")
    print("  • Coordinator queues commands and executes picks")
    print("\nUsage:")
    print("  1. Press Enter to start speech recording, Enter again to stop:")
    print("     Examples: 'pick up the cup', 'grab the ball', 'pick the bottle'")
    print("  2. Also supported: 'open gripper', 'close gripper', 'go home', 'stop'")
    print("  3. Press Ctrl+C to stop")
    print("  4. Live preview at http://127.0.0.1:8000")
    print("\n" + "=" * 70 + "\n")
    
    threads = []
    
    try:
        # Launch speech module
        if not args.vision_only:
            _launch_module_thread(threads, run_speech_module, "SpeechRecognition", "Speech recognition")
        
        # Launch vision module
        if not args.speech_only:
            _launch_module_thread(threads, run_vision_module, "ComputerVision", "Computer vision")
        
        print("\n✓ All modules initialized. System ready!")
        print("=" * 70 + "\n")
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  System shutdown initiated...")
        logger.info("Waiting for modules to finish...")
        
        # Threads are daemon=False, so they'll exit when main exits
        # Give them a moment to clean up
        for thread in threads:
            thread.join(timeout=5.0)
        
        print("✓ All modules stopped")
        print("Goodbye!\n")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
