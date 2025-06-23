"""
Voice Authenticator - Phase 2

This module implements voice authentication (speaker verification) using machine learning.
It extracts audio features and uses them to verify if the speaker is the authorized user.

Professional ML practices demonstrated:
- Feature extraction using MFCC and spectral features
- Model persistence and loading
- Confidence scoring and thresholds
- Training data management
"""

import os
import pickle
import numpy as np
import librosa
import logging
from typing import Optional, Tuple, List, Dict, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tempfile
import wave


class AudioFeatureExtractor:
    """
    Extracts features from audio for voice authentication
    
    Uses industry-standard audio features:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Spectral features (centroid, bandwidth, rolloff)
    - Zero crossing rate
    - Chroma features
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive audio features from raw audio data
        
        Args:
            audio_data: Raw audio samples as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            features = []
            
            # 1. MFCC features (most important for speaker recognition)
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13,
                hop_length=512,
                n_fft=2048
            )
            # Use statistics of MFCCs over time
            features.extend(np.mean(mfccs, axis=1))  # Mean
            features.extend(np.std(mfccs, axis=1))   # Standard deviation
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 4. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.array([])


class VoiceAuthenticator:
    """
    Voice authentication system using machine learning
    
    This implements speaker verification using:
    - Audio feature extraction
    - SVM classification with confidence scoring
    - Training data management
    - Model persistence
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ML components
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config.get('sample_rate', 16000)
        )
        self.scaler = StandardScaler()
        self.classifier = SVC(probability=True, kernel='rbf')
        
        # Model state
        self.is_trained = False
        self.model_path = config.get('model_path', 'models/voice_model.pkl')
        self.threshold = config.get('recognition_threshold', 0.85)
        
        # Training data storage
        self.training_samples: List[np.ndarray] = []
        self.training_labels: List[int] = []  # 1 for authorized user, 0 for others
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def start_enrollment(self, num_samples: int = 10) -> bool:
        """
        Start voice enrollment process
        
        Args:
            num_samples: Number of voice samples to collect
            
        Returns:
            True if enrollment started successfully
        """
        self.logger.info(f"Starting voice enrollment - collecting {num_samples} samples")
        self.training_samples.clear()
        self.training_labels.clear()
        return True
    
    def add_enrollment_sample(self, audio_data: np.ndarray, is_authorized: bool = True) -> bool:
        """
        Add a voice sample for training
        
        Args:
            audio_data: Raw audio data
            is_authorized: True if this is the authorized user
            
        Returns:
            True if sample added successfully
        """
        try:
            features = self.feature_extractor.extract_features(audio_data)
            if features.size == 0:
                self.logger.error("Failed to extract features from audio sample")
                return False
            
            self.training_samples.append(features)
            self.training_labels.append(1 if is_authorized else 0)
            
            self.logger.info(f"Added training sample {len(self.training_samples)} "
                           f"({'authorized' if is_authorized else 'unauthorized'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding enrollment sample: {e}")
            return False
    
    def complete_enrollment(self, add_negative_samples: bool = True) -> bool:
        """
        Complete enrollment and train the model
        
        Args:
            add_negative_samples: Whether to generate negative samples
            
        Returns:
            True if training completed successfully
        """
        try:
            if len(self.training_samples) < 3:
                self.logger.error("Need at least 3 samples for training")
                return False
            
            # Add some negative samples (noise, different speakers, etc.)
            if add_negative_samples:
                self._add_negative_samples()
            
            # Prepare training data
            X = np.array(self.training_samples)
            y = np.array(self.training_labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.classifier.fit(X_scaled, y)
            
            # Calculate training accuracy
            y_pred = self.classifier.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            
            self.is_trained = True
            self.logger.info(f"Voice model trained successfully! Accuracy: {accuracy:.2%}")
            
            # Save model
            self.save_model()
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing enrollment: {e}")
            return False
    
    def authenticate(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Authenticate a voice sample
        
        Args:
            audio_data: Raw audio data to authenticate
            
        Returns:
            Tuple of (is_authenticated, confidence_score)
        """
        try:
            if not self.is_trained:
                self.logger.warning("Voice model not trained - cannot authenticate")
                return False, 0.0
            
            # Extract features
            features = self.feature_extractor.extract_features(audio_data)
            if features.size == 0:
                return False, 0.0
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            # Get confidence for authorized user (class 1)
            confidence = probabilities[1] if len(probabilities) > 1 else 0.0
            
            # Check against threshold
            is_authenticated = confidence >= self.threshold
            
            self.logger.info(f"Voice authentication: {'PASSED' if is_authenticated else 'FAILED'} "
                           f"(confidence: {confidence:.2%}, threshold: {self.threshold:.2%})")
            
            return is_authenticated, confidence
            
        except Exception as e:
            self.logger.error(f"Error during authentication: {e}")
            return False, 0.0
    
    def save_model(self) -> bool:
        """Save the trained model to disk"""
        try:
            model_data = {
                'scaler': self.scaler,
                'classifier': self.classifier,
                'is_trained': self.is_trained,
                'threshold': self.threshold,
                'feature_extractor_config': {
                    'sample_rate': self.feature_extractor.sample_rate
                }
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Voice model saved to {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load a trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.info("No existing voice model found")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.classifier = model_data['classifier']
            self.is_trained = model_data['is_trained']
            self.threshold = model_data.get('threshold', self.threshold)
            
            self.logger.info(f"Voice model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def _add_negative_samples(self) -> None:
        """Add negative samples for better classification"""
        try:
            # Generate some noise samples as negative examples
            for i in range(3):
                # Create noise sample
                noise = np.random.normal(0, 0.1, 16000)  # 1 second of noise
                features = self.feature_extractor.extract_features(noise)
                if features.size > 0:
                    self.training_samples.append(features)
                    self.training_labels.append(0)
            
            self.logger.info("Added negative samples for training")
            
        except Exception as e:
            self.logger.error(f"Error adding negative samples: {e}")
    
    def get_enrollment_progress(self) -> Dict[str, Any]:
        """Get current enrollment progress"""
        return {
            'samples_collected': len(self.training_samples),
            'is_trained': self.is_trained,
            'model_exists': os.path.exists(self.model_path)
        }
