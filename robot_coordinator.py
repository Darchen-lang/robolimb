"""
ROBOT COORDINATOR - Unified Interface for Speech + Vision Integration
Bridges speech recognition and computer vision for voice-controlled object picking.
"""

import os
import sys
import queue
import threading
import logging
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Commands that flow between speech and vision modules
@dataclass
class RobotCommand:
    """Command from speech -> vision system."""
    command_type: str  # "pick", "home", "stop", "status"
    target_object: Optional[str] = None  # object class name to pick (e.g., "cup", "ball")
    confidence_threshold: float = 0.70
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"RobotCommand(type={self.command_type}, target={self.target_object})"


class RobotCoordinator:
    """
    Central coordinator that manages communication between speech recognition
    and computer vision modules.
    """
    def __init__(self):
        # Command queue: speech -> vision
        self.command_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # Status queue: vision -> speech/UI
        self.status_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # Current state
        self.current_target: Optional[str] = None
        self.is_picking: bool = False
        self.last_status: Optional[str] = None
        self.lock = threading.Lock()
        
        logger.info("RobotCoordinator initialized")

    def queue_pick_command(self, target_object: str, confidence: float = 0.70) -> bool:
        """
        Queue a pick command for a specific object.
        Returns True if queued successfully, False if queue is full.
        """
        cmd = RobotCommand(
            command_type="pick",
            target_object=target_object.lower().strip(),
            confidence_threshold=confidence
        )
        try:
            self.command_queue.put_nowait(cmd)
            with self.lock:
                self.current_target = target_object
            logger.info(f"Queued pick command for: {target_object}")
            return True
        except queue.Full:
            logger.warning(f"Command queue full, dropping pick request for {target_object}")
            return False

    def queue_command(self, command_type: str) -> bool:
        """Queue a generic command (home, stop, status)."""
        cmd = RobotCommand(command_type=command_type)
        try:
            self.command_queue.put_nowait(cmd)
            logger.info(f"Queued command: {command_type}")
            return True
        except queue.Full:
            logger.warning(f"Command queue full, dropping {command_type} request")
            return False

    def get_next_command(self, timeout: float = 0.5) -> Optional[RobotCommand]:
        """Get the next command from the queue (blocking with timeout)."""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def post_status(self, status: str, details: Optional[Dict] = None) -> bool:
        """Post status update from vision to listening modules."""
        try:
            msg = {
                "status": status,
                "timestamp": time.time(),
                "details": details or {}
            }
            self.status_queue.put_nowait(msg)
            with self.lock:
                self.last_status = status
            logger.debug(f"Posted status: {status}")
            return True
        except queue.Full:
            logger.warning(f"Status queue full, dropping update: {status}")
            return False

    def get_status(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get latest status from vision module."""
        try:
            return self.status_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_picking_state(self, is_picking: bool):
        """Update picking state."""
        with self.lock:
            self.is_picking = is_picking


# Global coordinator instance
coordinator = RobotCoordinator()


def main():
    """Test the coordinator."""
    print("Robot Coordinator Test")
    print("=" * 50)
    
    # Simulate command
    coordinator.queue_pick_command("cup", 0.75)
    cmd = coordinator.get_next_command(timeout=2)
    print(f"Retrieved command: {cmd}")
    
    # Simulate status
    coordinator.post_status("picking_in_progress", {"target": "cup"})
    status = coordinator.get_status(timeout=2)
    print(f"Retrieved status: {status}")


if __name__ == "__main__":
    main()
