"""
Voice Activator Engine - Enhanced with Voice Authentication

This is the main orchestrator of the entire system.
Phase 2: Added voice authentication integration.

Professional developers use this pattern called "Engine" or "Controller"
to coordinate between different subsystems.
"""

import logging
import threading
import time
from typing import Optional
from enum import Enum

from core.config import ConfigManager
from voice.processor import VoiceProcessor
from voice.enrollment import VoiceEnrollmentManager


class SystemState(Enum):
    """System states - professional developers use enums for state management"""
    STARTING = "starting"
    LISTENING = "listening"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    ENROLLING = "enrolling"
    LOCKED = "locked"
    SHUTTING_DOWN = "shutting_down"


class VoiceActivatorEngine:
    """
    Main system engine that coordinates all components
    
    This demonstrates the "Facade" design pattern - providing a simple
    interface to a complex subsystem.    """
    
    def __init__(self, config: ConfigManager):
        """Initialize the voice activator engine"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = SystemState.STARTING
        self._running = False
        self._main_thread: Optional[threading.Thread] = None
        
        # Initialize voice processor with enhanced config
        voice_config = {
            'activation_phrase': config.get('voice_auth.activation_phrase', 'hey jarvis'),
            'recognition_threshold': config.get('voice_auth.recognition_threshold', 0.85),
            'sample_rate': config.get('voice_auth.sample_rate', 16000),
            'chunk_size': config.get('voice_auth.chunk_size', 1024),
            'model_path': config.get('voice_auth.model_path', 'models/voice_model.pkl'),
        }
        
        self.voice_processor = VoiceProcessor(voice_config)
        
        # Initialize enrollment manager
        enrollment_config = {
            'training_samples': config.get('voice_auth.training_samples', 10),
        }
        self.enrollment_manager = VoiceEnrollmentManager(
            self.voice_processor.get_authenticator(), 
            enrollment_config
        )
        
        # Set up voice processor callbacks
        self.voice_processor.on_activation_phrase = self._handle_activation_phrase
        self.voice_processor.on_command = self._handle_voice_command
        self.voice_processor.on_error = self._handle_voice_error
        self.voice_processor.on_authentication_required = self._handle_authentication_required
        self.voice_processor.on_authentication_result = self._handle_authentication_result
        
        # Set up enrollment callbacks
        self.enrollment_manager.on_state_changed = self._handle_enrollment_state_changed
        self.enrollment_manager.on_progress_updated = self._handle_enrollment_progress
        self.enrollment_manager.on_sample_needed = self._handle_enrollment_sample_needed
        self.enrollment_manager.on_enrollment_complete = self._handle_enrollment_complete
        self.enrollment_manager.on_error = self._handle_enrollment_error
        
        # Component placeholders - we'll implement these next
        self.ui_overlay = None
        self.security_manager = None
        self.input_controller = None
        
        # Authentication state
        self.failed_attempts = 0
        self.max_failed_attempts = config.get('security.max_failed_attempts', 3)
        self.lockout_duration = config.get('security.lockout_duration', 300)
        self.last_failed_time = 0
        
        self.logger.info("Voice Activator Engine initialized with authentication")
    
    def start(self) -> None:
        """
        Start the main system        
        Professional practice: Non-blocking start with proper thread management
        """
        if self._running:
            self.logger.warning("Engine already running")
            return
        
        self._running = True
        self.state = SystemState.LISTENING
        
        # Start voice processor
        if not self.voice_processor.start_listening():
            self.logger.error("Failed to start voice processor")
            self._running = False
            return
        
        # Start main loop in separate thread
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        
        self.logger.info("Voice Activator Engine started")        
        # Keep main thread alive
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self) -> None:
        """Stop the system gracefully"""
        self.logger.info("Shutting down Voice Activator Engine...")
        self.state = SystemState.SHUTTING_DOWN
        self._running = False
        
        # Stop voice processor
        if self.voice_processor:
            self.voice_processor.stop_listening()
        
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=5.0)
        
        self.logger.info("Voice Activator Engine stopped")
    
    def _main_loop(self) -> None:
        """
        Main system loop
        
        This is where the magic happens - the main state machine
        """
        self.logger.info("Starting main system loop")
        
        while self._running:
            try:
                if self.state == SystemState.LISTENING:
                    self._handle_listening_state()
                elif self.state == SystemState.AUTHENTICATING:
                    self._handle_authentication_state()
                elif self.state == SystemState.ENROLLING:
                    self._handle_enrollment_state()
                elif self.state == SystemState.AUTHENTICATED:
                    self._handle_authenticated_state()
                elif self.state == SystemState.LOCKED:
                    self._handle_locked_state()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)  # Prevent rapid error loops
    def _handle_listening_state(self) -> None:
        """Handle the listening state - waiting for activation phrase"""
        # TODO: Implement voice listening
        # For now, just demonstrate the structure
        time.sleep(0.1)
    def _handle_authentication_state(self) -> None:
        """Handle voice authentication state"""
        try:
            # Check if we're in lockout
            if self._is_locked_out():
                self.logger.warning("System locked out due to failed attempts")
                self.state = SystemState.LOCKED
                return
            
            # Perform voice authentication
            is_authenticated, confidence = self.voice_processor.authenticate_voice()
            
            if is_authenticated:
                self.logger.info(f"Voice authentication successful! (confidence: {confidence:.2%})")
                self.failed_attempts = 0
                self.state = SystemState.AUTHENTICATED
            else:
                self.failed_attempts += 1
                self.last_failed_time = time.time()
                self.logger.warning(f"Voice authentication failed. Attempts: {self.failed_attempts}/{self.max_failed_attempts}")
                
                if self.failed_attempts >= self.max_failed_attempts:
                    self.logger.error("Maximum authentication attempts exceeded - entering lockout")
                    self.state = SystemState.LOCKED
                else:
                    self.state = SystemState.LISTENING
                    
        except Exception as e:
            self.logger.error(f"Error during authentication handling: {e}", exc_info=True)
            self.state = SystemState.LISTENING
    
    def _handle_enrollment_state(self) -> None:
        """Handle voice enrollment state"""
        # Enrollment is managed by the enrollment manager
        # We just wait for it to complete or fail
        time.sleep(0.1)
    
    def _handle_authenticated_state(self) -> None:
        """Handle authenticated state - process commands"""
        # TODO: Implement command processing
        time.sleep(0.1)
    
    def _handle_locked_state(self) -> None:
        """Handle locked state - block input, show overlay"""
        # TODO: Implement lock screen
        time.sleep(0.1)
    
    def get_state(self) -> SystemState:
        """Get current system state"""
        return self.state
    
    def is_running(self) -> bool:
        """Check if engine is running"""
        return self._running
    
    def _handle_activation_phrase(self, text: str) -> None:
        """Handle activation phrase detection"""
        self.logger.info(f"Activation phrase detected: '{text}'")
        
        # Check if voice model is trained
        if not self.voice_processor.is_voice_model_trained():
            self.logger.warning("Voice model not trained - starting enrollment")
            self.start_voice_enrollment()
        else:
            self.state = SystemState.AUTHENTICATING
    
    def _handle_voice_command(self, text: str) -> None:
        """Handle voice commands"""
        self.logger.info(f"Voice command: '{text}'")
        
        # Only process commands if authenticated
        if self.state != SystemState.AUTHENTICATED:
            self.logger.warning("Voice command ignored - not authenticated")
            return
        
        # Check for known commands
        if "unlock computer" in text or "log me in" in text:
            self.logger.info("Login command detected")
            # TODO: Add login logic
        elif "lock down" in text or "logout" in text:
            self.logger.info("Logout command detected")
            self.state = SystemState.LOCKED
        elif "start enrollment" in text or "retrain voice" in text:
            self.logger.info("Voice enrollment command detected")
            self.start_voice_enrollment()
        else:
            self.logger.debug(f"Unknown command: '{text}'")
    
    def _handle_voice_error(self, error: str) -> None:
        """Handle voice processing errors"""
        self.logger.error(f"Voice processing error: {error}")
        # TODO: Add error recovery logic
    
    def _handle_authentication_required(self) -> None:
        """Handle authentication required event"""
        self.logger.info("Voice authentication required")
        # TODO: Update UI to show authentication prompt
    
    def _handle_authentication_result(self, is_authenticated: bool, confidence: float) -> None:
        """Handle authentication result"""
        if is_authenticated:
            self.logger.info(f"Voice authentication successful (confidence: {confidence:.2%})")
        else:
            self.logger.warning(f"Voice authentication failed (confidence: {confidence:.2%})")
    
    def start_voice_enrollment(self) -> bool:
        """Start the voice enrollment process"""
        try:
            self.state = SystemState.ENROLLING
            return self.enrollment_manager.start_enrollment()
        except Exception as e:
            self.logger.error(f"Error starting voice enrollment: {e}")
            return False
    
    def get_enrollment_progress(self) -> dict:
        """Get current enrollment progress"""
        return self.enrollment_manager.get_progress()
    
    def get_authentication_status(self) -> dict:
        """Get current authentication status"""
        return {
            'is_trained': self.voice_processor.is_voice_model_trained(),
            'failed_attempts': self.failed_attempts,
            'max_attempts': self.max_failed_attempts,
            'is_locked_out': self._is_locked_out(),
            'state': self.state.value
        }
    
    def _is_locked_out(self) -> bool:
        """Check if system is currently locked out"""
        if self.failed_attempts < self.max_failed_attempts:
            return False
        
        # Check if lockout period has expired
        time_since_last_failure = time.time() - self.last_failed_time
        return time_since_last_failure < self.lockout_duration
    
    # Enrollment callback handlers
    def _handle_enrollment_state_changed(self, state):
        """Handle enrollment state changes"""
        self.logger.info(f"Enrollment state changed: {state.value}")
        
        if state.value == 'completed':
            self.state = SystemState.LISTENING
        elif state.value == 'failed':
            self.state = SystemState.LISTENING
    
    def _handle_enrollment_progress(self, progress):
        """Handle enrollment progress updates"""
        self.logger.info(f"Enrollment progress: {progress['current_sample']}/{progress['target_samples']}")
    
    def _handle_enrollment_sample_needed(self, phrase, sample_num, total_samples):
        """Handle when enrollment needs a sample"""
        self.logger.info(f"Enrollment sample {sample_num}/{total_samples} needed: '{phrase}'")
    
    def _handle_enrollment_complete(self, success, message):
        """Handle enrollment completion"""
        if success:
            self.logger.info(f"Enrollment completed: {message}")
            self.state = SystemState.LISTENING
        else:
            self.logger.error(f"Enrollment failed: {message}")
            self.state = SystemState.LISTENING
    
    def _handle_enrollment_error(self, error):
        """Handle enrollment errors"""
        self.logger.error(f"Enrollment error: {error}")
        self.state = SystemState.LISTENING
    
    # Authentication callback handlers
    def _handle_authentication_required(self):
        """Handle when authentication is required"""
        self.logger.info("Voice authentication required")
    def _handle_authentication_result(self, is_authenticated, confidence):
        """Handle authentication results"""
        if is_authenticated:
            self.logger.info(f"Authentication successful (confidence: {confidence:.2%})")
        else:
            self.logger.warning(f"Authentication failed (confidence: {confidence:.2%})")
    
    def start_voice_enrollment(self) -> bool:
        """Start the voice enrollment process"""
        try:
            self.state = SystemState.ENROLLING
            return self.enrollment_manager.start_enrollment()
        except Exception as e:
            self.logger.error(f"Error starting voice enrollment: {e}")
            return False
    
    def get_enrollment_progress(self) -> dict:
        """Get current enrollment progress"""
        return self.enrollment_manager.get_progress()
    
    def get_authentication_status(self) -> dict:
        """Get current authentication status"""
        return {
            'is_trained': self.voice_processor.is_voice_model_trained(),
            'failed_attempts': self.failed_attempts,
            'max_attempts': self.max_failed_attempts,
            'is_locked_out': self._is_locked_out(),
            'state': self.state.value
        }
    
    def _is_locked_out(self) -> bool:
        """Check if system is currently locked out"""
        if self.failed_attempts < self.max_failed_attempts:
            return False
        
        # Check if lockout period has expired
        time_since_last_failure = time.time() - self.last_failed_time
        return time_since_last_failure < self.lockout_duration
