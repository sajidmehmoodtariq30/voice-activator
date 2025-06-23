"""
Voice Enrollment Manager

This module manages the voice enrollment process, making it easy for users
to train the voice authentication system.

Professional UX practices:
- Clear step-by-step enrollment
- Progress tracking
- Error recovery
- User-friendly prompts
"""

import logging
import time
import threading
from typing import Optional, Callable, List
from enum import Enum
import speech_recognition as sr
import numpy as np

from voice.authenticator import VoiceAuthenticator


class EnrollmentState(Enum):
    """Enrollment process states"""
    IDLE = "idle"
    STARTING = "starting"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceEnrollmentManager:
    """
    Manages the voice enrollment process
    
    This provides a user-friendly interface for voice enrollment:
    - Step-by-step guidance
    - Progress tracking
    - Quality validation
    - Error handling
    """
    
    def __init__(self, authenticator: VoiceAuthenticator, config: dict):
        self.authenticator = authenticator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enrollment state
        self.state = EnrollmentState.IDLE
        self.current_sample = 0
        self.target_samples = config.get('training_samples', 10)
        self.min_duration = 2.0  # Minimum seconds per sample
        self.max_duration = 5.0  # Maximum seconds per sample
        
        # Audio capture
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Callbacks for UI updates
        self.on_state_changed: Optional[Callable] = None
        self.on_progress_updated: Optional[Callable] = None
        self.on_sample_needed: Optional[Callable] = None
        self.on_enrollment_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Enrollment phrases - what the user should say
        self.enrollment_phrases = [
            "Hey Jarvis, this is my voice for authentication",
            "Hello computer, remember my voice pattern",
            "Access granted to authorized user only",
            "My voice is the key to this system",
            "Voice authentication sample number {}",
            "This is my unique voice signature",
            "Computer, learn to recognize my speech",
            "Biometric voice sample for security",
            "Only my voice should unlock this system",
            "Voice identification sample complete"
        ]
    
    def start_enrollment(self) -> bool:
        """
        Start the voice enrollment process
        
        Returns:
            True if enrollment started successfully
        """
        try:
            if self.state != EnrollmentState.IDLE:
                self.logger.warning("Enrollment already in progress")
                return False
            
            # Initialize microphone
            try:
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info("Microphone calibrated for ambient noise")
            except Exception as e:
                self.logger.error(f"Failed to initialize microphone: {e}")
                self._set_state(EnrollmentState.FAILED)
                self._call_error_callback(f"Microphone initialization failed: {e}")
                return False
            
            # Start enrollment
            self._set_state(EnrollmentState.STARTING)
            self.current_sample = 0
            
            # Initialize authenticator enrollment
            if not self.authenticator.start_enrollment(self.target_samples):
                self.logger.error("Failed to start authenticator enrollment")
                self._set_state(EnrollmentState.FAILED)
                return False
            
            self.logger.info(f"Voice enrollment started - need {self.target_samples} samples")
            self._set_state(EnrollmentState.COLLECTING)
            self._request_next_sample()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting enrollment: {e}")
            self._set_state(EnrollmentState.FAILED)
            self._call_error_callback(str(e))
            return False
    
    def capture_sample(self) -> bool:
        """
        Capture a single voice sample
        
        Returns:
            True if sample captured successfully
        """
        try:
            if self.state != EnrollmentState.COLLECTING:
                self.logger.warning("Not in collecting state")
                return False
            
            self.logger.info(f"Capturing sample {self.current_sample + 1}/{self.target_samples}")
            
            # Get the phrase for this sample
            phrase = self.enrollment_phrases[self.current_sample % len(self.enrollment_phrases)]
            if "{}" in phrase:
                phrase = phrase.format(self.current_sample + 1)
              # Notify UI to prompt user
            if self.on_sample_needed:
                self.on_sample_needed(phrase, self.current_sample + 1, self.target_samples)
            else:
                # Log the prompt for console users
                self.logger.info(f"📣 ENROLLMENT SAMPLE {self.current_sample + 1}/{self.target_samples}")
                self.logger.info(f"🎤 Please say: '{phrase}'")
                self.logger.info("🔴 Recording will start in 3 seconds...")
              # Capture audio with retry mechanism
            max_retries = 3
            audio_data = None
            
            for attempt in range(max_retries):
                self.logger.info(f"🎯 Capture attempt {attempt + 1}/{max_retries}")
                audio_data = self._capture_audio_sample()
                
                if audio_data is not None:
                    break
                elif attempt < max_retries - 1:
                    self.logger.warning(f"⚠️  Capture attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)  # Brief pause before retry
                else:
                    self.logger.error("❌ All capture attempts failed")
            
            if audio_data is None:
                self.logger.error("Failed to capture audio sample after retries")
                self._set_state(EnrollmentState.COLLECTING)  # Stay in collecting state
                return False
            
            # Validate sample quality
            if not self._validate_sample_quality(audio_data):
                self.logger.warning("Sample quality too low, skipping")
                return False
            
            # Add to authenticator
            self._set_state(EnrollmentState.PROCESSING)
            if not self.authenticator.add_enrollment_sample(audio_data, is_authorized=True):
                self.logger.error("Failed to add sample to authenticator")
                self._set_state(EnrollmentState.COLLECTING)
                return False
            
            # Update progress
            self.current_sample += 1
            self._update_progress()
            
            # Check if we're done
            if self.current_sample >= self.target_samples:
                self._complete_enrollment()
            else:
                self._set_state(EnrollmentState.COLLECTING)
                self._request_next_sample()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing sample: {e}")
            self._set_state(EnrollmentState.COLLECTING)
            return False
    
    def cancel_enrollment(self) -> None:
        """Cancel the enrollment process"""
        self.logger.info("Enrollment cancelled by user")
        self._set_state(EnrollmentState.IDLE)
        self.current_sample = 0
    
    def get_progress(self) -> dict:
        """Get current enrollment progress"""
        return {
            'state': self.state.value,
            'current_sample': self.current_sample,
            'target_samples': self.target_samples,
            'progress_percent': (self.current_sample / self.target_samples) * 100 if self.target_samples > 0 else 0
        }
    
    def _capture_audio_sample(self) -> Optional[np.ndarray]:
        """
        Capture a single audio sample from microphone
        
        Returns:
            Audio data as numpy array, or None if failed
        """
        try:
            # Give user time to prepare
            self.logger.info("🎤 Get ready to speak...")
            time.sleep(3.0)
            
            self.logger.info("🔴 RECORDING NOW! Speak clearly...")
            
            # Record for specified duration with robust error handling
            try:
                with self.microphone as source:
                    # Record for specified duration
                    audio = self.recognizer.listen(
                        source, 
                        timeout=15.0,  # Increased timeout
                        phrase_time_limit=self.max_duration
                    )
                
                self.logger.info("🎵 Audio captured, processing...")
                
            except sr.WaitTimeoutError:
                self.logger.warning("⚠️  No speech detected within timeout period")
                return None
            except Exception as audio_error:
                self.logger.error(f"❌ Audio recording error: {audio_error}")
                return None
            
            # Convert to numpy array with error handling
            try:
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                self.logger.info(f"✅ Captured {len(audio_data)} samples ({len(audio_data)/16000:.1f}s)")
                return audio_data
                
            except Exception as convert_error:
                self.logger.error(f"❌ Audio conversion error: {convert_error}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in audio capture: {e}", exc_info=True)
            return None
    
    def _validate_sample_quality(self, audio_data: np.ndarray) -> bool:
        """
        Validate the quality of an audio sample
        
        Args:
            audio_data: Audio data to validate
            
        Returns:
            True if sample quality is acceptable
        """
        try:
            # Check duration
            duration = len(audio_data) / 16000.0  # Assuming 16kHz sample rate
            if duration < self.min_duration:
                self.logger.warning(f"Sample too short: {duration:.1f}s < {self.min_duration}s")
                return False
            
            # Check amplitude (not too quiet)
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude < 0.01:  # Very quiet
                self.logger.warning(f"Sample too quiet: max amplitude {max_amplitude:.3f}")
                return False
            
            # Check for clipping (not too loud)
            if max_amplitude > 0.95:
                self.logger.warning(f"Sample may be clipped: max amplitude {max_amplitude:.3f}")
                # Don't reject, just warn
            
            # Check RMS energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            if rms_energy < 0.005:
                self.logger.warning(f"Sample has low energy: RMS {rms_energy:.4f}")
                return False
            
            self.logger.info(f"Sample quality OK: duration={duration:.1f}s, "
                           f"max_amp={max_amplitude:.3f}, rms={rms_energy:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating sample: {e}")
            return False
    
    def _complete_enrollment(self) -> None:
        """Complete the enrollment process"""
        try:
            self.logger.info("Completing voice enrollment...")
            self._set_state(EnrollmentState.PROCESSING)
            
            # Train the model
            if self.authenticator.complete_enrollment():
                self.logger.info("Voice enrollment completed successfully!")
                self._set_state(EnrollmentState.COMPLETED)
                
                if self.on_enrollment_complete:
                    self.on_enrollment_complete(True, "Enrollment completed successfully")
            else:
                self.logger.error("Failed to complete voice enrollment")
                self._set_state(EnrollmentState.FAILED)
                
                if self.on_enrollment_complete:
                    self.on_enrollment_complete(False, "Failed to train voice model")
            
        except Exception as e:
            self.logger.error(f"Error completing enrollment: {e}")
            self._set_state(EnrollmentState.FAILED)
            self._call_error_callback(str(e))
    
    def _set_state(self, new_state: EnrollmentState) -> None:
        """Update enrollment state and notify callbacks"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.logger.debug(f"Enrollment state: {old_state.value} -> {new_state.value}")
            
            if self.on_state_changed:
                self.on_state_changed(new_state)
    
    def _update_progress(self) -> None:
        """Update progress and notify callbacks"""
        if self.on_progress_updated:
            progress = self.get_progress()
            self.on_progress_updated(progress)
    def _request_next_sample(self) -> None:
        """Request the next sample from the user"""
        # Automatically start capturing the next sample after a short delay
        def start_capture():
            if self.state == EnrollmentState.COLLECTING:
                self.capture_sample()
        
        threading.Timer(2.0, start_capture).start()
    
    def _call_error_callback(self, error_message: str) -> None:
        """Call error callback if set"""
        if self.on_error:
            self.on_error(error_message)
