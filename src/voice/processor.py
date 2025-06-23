"""
Voice Processor - Enhanced with Authentication

This module handles all voice recognition and audio processing.
Phase 2: Added voice authentication capabilities.

Professional developers start with simple, working code and gradually add complexity.
"""

import speech_recognition as sr
import logging
import threading
import time
import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum

from voice.authenticator import VoiceAuthenticator


class VoiceProcessorState(Enum):
    """Voice processor states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    AUTHENTICATING = "authenticating"
    ERROR = "error"


class VoiceProcessor:
    """
    Handles voice recognition and processing with authentication
    
    Enhanced with Phase 2 features:
    - Voice authentication integration
    - Audio capture for biometric analysis
    - Authentication callbacks
    
    This demonstrates professional practices:
    - Clean interface design
    - Error handling
    - State management
    - Callback patterns
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = VoiceProcessorState.IDLE
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._listening_thread: Optional[threading.Thread] = None
        self._stop_listening = False
        
        # Initialize voice authenticator
        auth_config = {
            'sample_rate': config.get('sample_rate', 16000),
            'model_path': config.get('model_path', 'models/voice_model.pkl'),
            'recognition_threshold': config.get('recognition_threshold', 0.85)
        }
        self.authenticator = VoiceAuthenticator(auth_config)
          # Callbacks for events
        self.on_activation_phrase: Optional[Callable] = None
        self.on_command: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # New authentication callbacks
        self.on_authentication_required: Optional[Callable] = None
        self.on_authentication_result: Optional[Callable] = None
        
        # Configuration from config
        self.activation_phrase = config.get('activation_phrase', 'hey jarvis').lower()
        self.recognition_threshold = config.get('recognition_threshold', 0.85)
        
        self.logger.info(f"Voice processor initialized with activation phrase: '{self.activation_phrase}'")
    
    def initialize_microphone(self) -> bool:
        """
        Initialize microphone for audio input
        
        Professional practice: Always handle hardware initialization gracefully
        """
        try:
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            self.logger.info("Adjusting for ambient noise... Please wait.")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
            self.logger.info("Microphone initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize microphone: {e}")
            self.state = VoiceProcessorState.ERROR
            return False
    
    def start_listening(self) -> bool:
        """
        Start continuous listening for voice input
        
        This is the core of our voice recognition system
        """
        if not self.microphone:
            if not self.initialize_microphone():
                return False
        
        if self._listening_thread and self._listening_thread.is_alive():
            self.logger.warning("Already listening")
            return True
        
        self._stop_listening = False
        self._listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listening_thread.start()
        
        self.state = VoiceProcessorState.LISTENING
        self.logger.info("Started listening for voice input")
        return True
    
    def stop_listening(self) -> None:
        """Stop listening for voice input"""
        self._stop_listening = True
        if self._listening_thread:
            self._listening_thread.join(timeout=5.0)
        
        self.state = VoiceProcessorState.IDLE
        self.logger.info("Stopped listening")
    
    def _listen_loop(self) -> None:
        """
        Main listening loop
        
        This runs in a separate thread and continuously listens for voice input
        """
        self.logger.info("Voice listening loop started")
        
        while not self._stop_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                self.state = VoiceProcessorState.PROCESSING
                
                # Recognize speech using Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    self.logger.debug(f"Recognized text: '{text}'")
                    
                    # Check for activation phrase
                    if self.activation_phrase in text:
                        self.logger.info(f"Activation phrase detected: '{text}'")
                        if self.on_activation_phrase:
                            self.on_activation_phrase(text)
                    else:
                        # Check if this might be a command (after activation)
                        if self.on_command:
                            self.on_command(text)
                
                except sr.UnknownValueError:
                    # Google Speech Recognition could not understand audio
                    self.logger.debug("Could not understand audio")
                
                except sr.RequestError as e:
                    self.logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    if self.on_error:
                        self.on_error(f"Speech recognition service error: {e}")
                
                self.state = VoiceProcessorState.LISTENING
                
            except sr.WaitTimeoutError:
                # No audio detected within timeout - this is normal
                pass
            
            except Exception as e:
                self.logger.error(f"Error in listening loop: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(f"Listening error: {e}")
                time.sleep(1)  # Prevent rapid error loops        
        self.logger.info("Voice listening loop stopped")
    
    def test_recognition(self) -> bool:
        """
        Test speech recognition functionality
        
        This is useful for development and debugging
        """
        if not self.microphone:
            if not self.initialize_microphone():
                return False
        
        try:
            self.logger.info("Say something for testing...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5)
            
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Test recognition result: '{text}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Test recognition failed: {e}")
            return False
    
    def authenticate_voice(self, duration_seconds: float = 3.0) -> Tuple[bool, float]:
        """
        Perform voice authentication
        
        Args:
            duration_seconds: How long to record for authentication
            
        Returns:
            Tuple of (is_authenticated, confidence_score)
        """
        try:
            if not self.microphone:
                if not self.initialize_microphone():
                    return False, 0.0
            
            self.state = VoiceProcessorState.AUTHENTICATING
            self.logger.info("Starting voice authentication...")
            
            # Notify that authentication is required
            if self.on_authentication_required:
                self.on_authentication_required()
            
            # Capture audio for authentication
            audio_data = self._capture_authentication_audio(duration_seconds)
            if audio_data is None:
                self.logger.error("Failed to capture audio for authentication")
                return False, 0.0
            
            # Perform authentication
            is_authenticated, confidence = self.authenticator.authenticate(audio_data)
            
            # Notify of result
            if self.on_authentication_result:
                self.on_authentication_result(is_authenticated, confidence)
            
            self.state = VoiceProcessorState.LISTENING
            return is_authenticated, confidence
            
        except Exception as e:
            self.logger.error(f"Error during voice authentication: {e}")
            self.state = VoiceProcessorState.LISTENING
            if self.on_error:
                self.on_error(f"Authentication error: {e}")
            return False, 0.0
    
    def _capture_authentication_audio(self, duration: float) -> Optional[np.ndarray]:
        """
        Capture audio specifically for authentication
        
        Args:
            duration: Duration in seconds to record
            
        Returns:
            Audio data as numpy array, or None if failed
        """
        try:
            self.logger.info(f"Recording for authentication ({duration}s)...")
            
            with self.microphone as source:
                # Record for the specified duration
                audio = self.recognizer.listen(
                    source, 
                    timeout=10.0,
                    phrase_time_limit=duration
                )
            
            # Convert to numpy array
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            self.logger.info(f"Captured {len(audio_data)} samples for authentication")
            return audio_data
            
        except sr.WaitTimeoutError:
            self.logger.warning("No speech detected for authentication")
            return None
        except Exception as e:
            self.logger.error(f"Error capturing authentication audio: {e}")
            return None
    
    def get_authenticator(self) -> VoiceAuthenticator:
        """Get the voice authenticator instance"""
        return self.authenticator
    
    def is_voice_model_trained(self) -> bool:
        """Check if voice authentication model is trained"""
        return self.authenticator.is_trained
