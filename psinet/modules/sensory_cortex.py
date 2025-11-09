## Developer: inkbytefo
## Modified: 2025-11-09

import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple
import threading
import queue
import time

logger = logging.getLogger(__name__)


class SensoryCortex:
    """
    Module 2: Sensory Cortex - Enhanced Vision and Audio Processing
    
    Processes raw sensory data (vision, audio) and converts it to spike trains
    that can be understood by the SNN. Includes feature extraction for
    more meaningful representations beyond simple pixel differences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Sensory Cortex with configuration.
        
        Args:
            config: Sensory configuration dictionary
        """
        self.config = config
        self.vision_enabled = config.get('vision', {}).get('enable', False)
        self.audio_enabled = config.get('audio', {}).get('enable', False)
        
        # Vision processing
        self.vision_processor = None
        if self.vision_enabled:
            self.vision_processor = VisionProcessor(config.get('vision', {}))
            logger.info("Vision processor initialized")
        
        # Audio processing
        self.audio_processor = None
        if self.audio_enabled:
            self.audio_processor = AudioProcessor(config.get('audio', {}))
            logger.info("Audio processor initialized")
        
        # Threading for real-time processing
        self.running = False
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
    def start(self):
        """Start real-time sensory processing."""
        if self.vision_enabled or self.audio_enabled:
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info("Sensory Cortex started")
    
    def stop(self):
        """Stop sensory processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Sensory Cortex stopped")
    
    def process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process sensory inputs and return spike trains.
        
        Args:
            inputs: Dictionary containing sensory data
            
        Returns:
            Dictionary mapping sensory modalities to spike trains
        """
        spike_trains = {}
        
        # Process vision
        if self.vision_enabled and 'vision' in inputs:
            if self.vision_processor:
                vision_spikes = self.vision_processor.process_frame(inputs['vision'])
                spike_trains['sensory_vision'] = vision_spikes
        
        # Process audio
        if self.audio_enabled and 'audio' in inputs:
            if self.audio_processor:
                audio_spikes = self.audio_processor.process_audio_chunk(inputs['audio'])
                spike_trains['sensory_audio'] = audio_spikes
        
        return spike_trains
    
    def _processing_loop(self):
        """Background processing loop for real-time sensory data."""
        while self.running:
            try:
                # Get sensory data from queue
                sensory_data = self.input_queue.get(timeout=0.1)
                
                # Process and store results
                spike_trains = self.process_inputs(sensory_data)
                
                if not self.output_queue.full():
                    self.output_queue.put(spike_trains)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sensory processing error: {e}")
    
    def get_latest_spikes(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the most recent spike trains."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None


class VisionProcessor:
    """Enhanced vision processing with feature extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vision processor.
        
        Args:
            config: Vision configuration
        """
        self.config = config
        self.target_size = tuple(config.get('target_size', [28, 28]))
        self.change_threshold = config.get('change_threshold', 30)
        self.max_rate = config.get('max_rate', 150.0)
        
        # Feature extraction settings
        self.feature_config = config.get('feature_extraction', {})
        self.enable_edges = self.feature_config.get('enable_edges', True)
        self.enable_corners = self.feature_config.get('enable_corners', True)
        self.enable_colors = self.feature_config.get('enable_colors', True)
        
        # Previous frame for change detection
        self.prev_frame = None
        self.prev_features = None
        
        # Initialize camera if available
        self.camera = None
        camera_id = config.get('camera_id', 0)
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if self.camera.isOpened():
                logger.info(f"Camera {camera_id} initialized")
            else:
                logger.warning(f"Camera {camera_id} not available")
                self.camera = None
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e}")
            self.camera = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and generate spike trains.
        
        Args:
            frame: Input frame (H x W x C)
            
        Returns:
            Spike train array
        """
        # Resize frame
        resized = cv2.resize(frame, self.target_size)
        
        # Convert to grayscale for basic processing
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        features = self._extract_features(resized, gray)
        
        # Calculate changes from previous frame
        if self.prev_frame is not None:
            # Pixel-based changes
            pixel_changes = np.abs(gray.astype(float) - self.prev_frame.astype(float))
            pixel_mask = pixel_changes > self.change_threshold
            
            # Feature-based changes
            feature_changes = np.abs(features - self.prev_features) if self.prev_features is not None else features
            
            # Combine pixel and feature changes
            combined_changes = pixel_mask.astype(float) * 0.5 + feature_changes * 0.5
        else:
            # First frame - use all features
            combined_changes = features
        
        # Convert to spike rates
        spike_rates = (combined_changes / 255.0) * self.max_rate
        
        # Update previous frame and features
        self.prev_frame = gray.copy()
        self.prev_features = features.copy()
        
        return spike_rates.flatten()
    
    def _extract_features(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Extract visual features from frame.
        
        Args:
            frame: Color frame
            gray: Grayscale frame
            
        Returns:
            Feature array
        """
        features = []
        
        # Basic intensity features
        features.append(gray.astype(float) / 255.0)
        
        # Edge detection
        if self.enable_edges:
            edges = cv2.Canny(gray, 50, 150)
            features.append(edges.astype(float) / 255.0)
        
        # Corner detection
        if self.enable_corners:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            corner_map = np.zeros_like(gray)
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    corner_map[int(y), int(x)] = 255
            features.append(corner_map.astype(float) / 255.0)
        
        # Color features
        if self.enable_colors:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Use hue and saturation channels
            color_features = hsv[:, :, :2].astype(float) / 255.0
            features.extend([color_features[:, :, 0], color_features[:, :, 1]])
        
        # Combine all features
        if len(features) > 1:
            combined = np.concatenate([f.flatten() for f in features])
        else:
            combined = features[0].flatten()
        
        # Normalize to [0, 1]
        combined = np.clip(combined, 0, 1)
        
        return combined
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera."""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()


class AudioProcessor:
    """Enhanced audio processing with frequency analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio processor.
        
        Args:
            config: Audio configuration
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 44100)
        self.chunk_size = config.get('chunk_size', 1024)
        self.freq_bins = config.get('freq_bins', 32)
        self.max_rate = config.get('max_rate', 150.0)
        
        # Audio processing state
        self.prev_spectrum = None
        
        # Initialize audio input if available
        self.audio_stream = None
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            logger.info("Audio input initialized")
        except ImportError:
            logger.warning("PyAudio not available - audio processing disabled")
        except Exception as e:
            logger.warning(f"Audio initialization failed: {e}")
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio chunk and generate spike trains.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Spike train array
        """
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(audio_data))
        windowed = audio_data * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        # Log-scale magnitude for better dynamic range
        log_magnitude = np.log1p(magnitude)
        
        # Bin the frequencies
        if len(log_magnitude) > self.freq_bins:
            # Simple binning - could use mel scale for more perceptually relevant bins
            bin_size = len(log_magnitude) // self.freq_bins
            binned = log_magnitude[:self.freq_bins * bin_size].reshape(self.freq_bins, bin_size).mean(axis=1)
        else:
            binned = log_magnitude
        
        # Calculate changes from previous spectrum
        if self.prev_spectrum is not None:
            changes = np.abs(binned - self.prev_spectrum)
        else:
            changes = binned
        
        # Normalize and convert to spike rates
        if np.max(changes) > 0:
            normalized_changes = changes / np.max(changes)
        else:
            normalized_changes = changes
        
        spike_rates = normalized_changes * self.max_rate
        
        # Update previous spectrum
        self.prev_spectrum = binned.copy()
        
        return spike_rates
    
    def read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of audio data."""
        if self.audio_stream is None:
            return None
        
        try:
            data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.float32)
            return audio_array
        except Exception as e:
            logger.error(f"Audio read error: {e}")
            return None
    
    def release(self):
        """Release audio resources."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'pa'):
            self.pa.terminate()