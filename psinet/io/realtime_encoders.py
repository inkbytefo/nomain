## Developer: inkbytefo
## Modified: 2025-11-08

import numpy as np
import logging
from typing import Optional, Tuple

# Optional imports for sensory processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except (ImportError, OSError):
    SD_AVAILABLE = False
    sd = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoToSpikes:
    """
    Real-time video encoder that converts camera frames to spike trains using
    Dynamic Vision Sensor (DVS) emulation logic.
    """
    
    def __init__(self, camera_id: int = 0, target_size: Tuple[int, int] = (28, 28),
                 change_threshold: int = 30, max_rate: float = 150.0):
        """
        Initialize video to spike encoder.
        
        Args:
            camera_id: Camera device ID (default: 0)
            target_size: Resize frames to this size (default: 28x28 for MNIST compatibility)
            change_threshold: Threshold for detecting significant changes
            max_rate: Maximum firing rate in Hz for changes
        """
        self.camera_id = camera_id
        self.target_size = target_size
        self.change_threshold = change_threshold
        self.max_rate = max_rate
        self.cap = None
        self.prev_frame = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize camera capture."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - video encoder disabled")
            return False
            
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            # Read first frame to initialize
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                return False
                
            # Convert to grayscale and resize
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, self.target_size)
            self.prev_frame = frame_resized
            self.is_initialized = True
            
            logger.info(f"VideoToSpikes initialized with camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            return False
    
    def get_spike_rates(self) -> np.ndarray:
        """
        Get current spike rates based on frame differences.
        
        Returns:
            np.ndarray: Flattened array of firing rates for each pixel
        """
        if not self.is_initialized or self.cap is None:
            logger.warning("VideoToSpikes not initialized")
            return np.zeros(self.target_size[0] * self.target_size[1])
            
        try:
            # Read current frame
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                return np.zeros(self.target_size[0] * self.target_size[1])
                
            # Convert to grayscale and resize
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, self.target_size)
            
            # Calculate absolute difference (DVS emulation)
            diff = cv2.absdiff(self.prev_frame, frame_resized)
            
            # Apply threshold to detect significant changes
            _, change_map = cv2.threshold(diff, self.change_threshold, 255, cv2.THRESH_BINARY)
            
            # Convert to Poisson firing rates
            # Normalize change map to 0-1 range
            change_normalized = change_map.astype(float) / 255.0
            
            # Convert to firing rates (high rate for change, zero for no change)
            spike_rates = change_normalized.flatten() * self.max_rate
            
            # Update previous frame
            self.prev_frame = frame_resized
            
            return spike_rates
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return np.zeros(self.target_size[0] * self.target_size[1])
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            logger.info("VideoToSpikes resources released")


class AudioToSpikes:
    """
    Real-time audio encoder that converts microphone input to spike trains
    using frequency domain analysis (cochleagram).
    """
    
    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024,
                 freq_bins: int = 28, max_rate: float = 150.0):
        """
        Initialize audio to spike encoder.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            chunk_size: Number of samples per audio chunk
            freq_bins: Number of frequency bins for analysis
            max_rate: Maximum firing rate in Hz
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.freq_bins = freq_bins
        self.max_rate = max_rate
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize audio stream."""
        if not SD_AVAILABLE:
            logger.warning("SoundDevice not available - audio encoder disabled")
            return False
            
        try:
            # Test audio device availability
            sd.check_input_settings(device=None, channels=1, dtype='float32',
                                   samplerate=self.sample_rate)
            self.is_initialized = True
            logger.info(f"AudioToSpikes initialized with sample_rate={self.sample_rate}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing audio: {e}")
            return False
    
    def get_spike_rates(self) -> np.ndarray:
        """
        Get current spike rates based on audio frequency analysis.
        
        Returns:
            np.ndarray: Array of firing rates for each frequency bin
        """
        if not self.is_initialized:
            logger.warning("AudioToSpikes not initialized")
            return np.zeros(self.freq_bins)
            
        try:
            # Record audio chunk
            audio_chunk = sd.rec(self.chunk_size, samplerate=self.sample_rate,
                                channels=1, dtype='float32')
            sd.wait()  # Wait for recording to complete
            
            # Extract mono channel
            audio_data = audio_chunk.flatten()
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(audio_data))
            windowed_audio = audio_data * window
            
            # Perform FFT to get frequency spectrum
            fft_result = np.fft.fft(windowed_audio)
            fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
            
            # Normalize frequency spectrum
            if np.max(fft_magnitude) > 0:
                fft_normalized = fft_magnitude / np.max(fft_magnitude)
            else:
                fft_normalized = fft_magnitude
            
            # Resample to desired number of frequency bins
            # Take log scale for more perceptually relevant representation
            fft_log = np.log1p(fft_normalized)
            
            # Downsample to target frequency bins
            freq_indices = np.linspace(0, len(fft_log) - 1, self.freq_bins, dtype=int)
            freq_energies = fft_log[freq_indices]
            
            # Normalize to 0-1 range
            if np.max(freq_energies) > 0:
                freq_normalized = freq_energies / np.max(freq_energies)
            else:
                freq_normalized = freq_energies
            
            # Convert to Poisson firing rates
            spike_rates = freq_normalized * self.max_rate
            
            return spike_rates
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return np.zeros(self.freq_bins)
    
    def get_combined_spike_rates(self, video_encoder: VideoToSpikes,
                                video_weight: float = 0.7, audio_weight: float = 0.3) -> np.ndarray:
        """
        Combine video and audio spike rates into a single input vector.
        
        Args:
            video_encoder: VideoToSpikes instance
            video_weight: Weight for video contribution (0-1)
            audio_weight: Weight for audio contribution (0-1)
            
        Returns:
            np.ndarray: Combined spike rates
        """
        video_rates = video_encoder.get_spike_rates()
        audio_rates = self.get_spike_rates()
        
        # Normalize weights
        total_weight = video_weight + audio_weight
        if total_weight > 0:
            video_weight /= total_weight
            audio_weight /= total_weight
        
        # Combine rates
        combined_size = len(video_rates) + len(audio_rates)
        combined_rates = np.zeros(combined_size)
        
        combined_rates[:len(video_rates)] = video_rates * video_weight
        combined_rates[len(video_rates):] = audio_rates * audio_weight
        
        return combined_rates


class RealtimeEncoder:
    """
    Unified interface for real-time sensory encoding.
    """
    
    def __init__(self, enable_video=True, enable_audio=True, config=None):
        """
        Initialize realtime encoder with configuration.
        
        Args:
            enable_video: Enable video encoder (default: True)
            enable_audio: Enable audio encoder (default: True)
            config: Optional configuration dictionary (for backward compatibility)
        """
        # Support both old config-based and new parameter-based initialization
        if config is not None:
            self.config = config
            enable_video = config.get('enable_video', True)
            enable_audio = config.get('enable_audio', True)
        else:
            self.config = {
                'enable_video': enable_video,
                'enable_audio': enable_audio,
                'video_weight': 0.7,
                'audio_weight': 0.3
            }
            
        self.video_encoder = None
        self.audio_encoder = None
        self.is_initialized = False
        
        # DÜZELTME: If both disabled, mark as initialized
        if not enable_video and not enable_audio:
            self.is_initialized = True
            logger.info("RealtimeEncoder: No sensors enabled - simulation mode")
            return
        
        # Initialize video encoder if enabled
        if enable_video and CV2_AVAILABLE:
            video_config = self.config.get('video', {}) if config else {}
            self.video_encoder = VideoToSpikes(
                camera_id=video_config.get('camera_id', 0),
                target_size=tuple(video_config.get('target_size', [28, 28])),
                change_threshold=video_config.get('change_threshold', 30),
                max_rate=video_config.get('max_rate', 150.0)
            )
        
        # Initialize audio encoder if enabled
        if enable_audio and SD_AVAILABLE:
            audio_config = self.config.get('audio', {}) if config else {}
            self.audio_encoder = AudioToSpikes(
                sample_rate=audio_config.get('sample_rate', 44100),
                chunk_size=audio_config.get('chunk_size', 1024),
                freq_bins=audio_config.get('freq_bins', 28),
                max_rate=audio_config.get('max_rate', 150.0)
            )
    
    def initialize(self) -> bool:
        """Initialize all enabled encoders."""
        # Check if any encoders are enabled
        has_enabled_encoders = self.video_encoder is not None or self.audio_encoder is not None
        
        if not has_enabled_encoders:
            logger.warning("No encoders enabled - running in simulation mode")
            self.is_initialized = True
            return True
        
        # Initialize enabled encoders
        success = True
        
        if self.video_encoder:
            video_success = self.video_encoder.initialize()
            success &= video_success
            if not video_success:
                logger.warning("Video encoder failed to initialize")
        
        if self.audio_encoder:
            audio_success = self.audio_encoder.initialize()
            success &= audio_success
            if not audio_success:
                logger.warning("Audio encoder failed to initialize")
        
        # Consider initialization successful if at least one encoder works
        if success:
            logger.info("RealtimeEncoder initialized successfully")
        else:
            logger.error("RealtimeEncoder initialization failed - no encoders available")
        
        self.is_initialized = success
        return success
    
    def get_spike_rates(self) -> np.ndarray:
        """
        Get combined spike rates from all enabled encoders.
        
        Returns:
            np.ndarray: Combined spike rates
        """
        if not self.is_initialized:
            logger.warning("RealtimeEncoder not initialized")
            return np.zeros(28 * 28)  # Default MNIST size
        
        try:
            if self.video_encoder and self.audio_encoder:
                # Combine both video and audio
                video_weight = self.config.get('video_weight', 0.7)
                audio_weight = self.config.get('audio_weight', 0.3)
                return self.audio_encoder.get_combined_spike_rates(
                    self.video_encoder, video_weight, audio_weight
                )
            elif self.video_encoder:
                # Video only
                return self.video_encoder.get_spike_rates()
            elif self.audio_encoder:
                # Audio only
                return self.audio_encoder.get_spike_rates()
            else:
                # No encoders enabled - return small random noise for simulation
                # This allows the neural network to remain active without sensory input
                return np.random.uniform(0, 1, 28 * 28) * 0.1  # Very low activity
                
        except Exception as e:
            logger.error(f"Error getting spike rates: {e}")
            return np.zeros(28 * 28)
    
    def get_combined_rates(self) -> np.ndarray:
        """
        Get combined spike rates from all enabled encoders.
        This method returns a fixed-size array (12000,) for compatibility with AGI.
        
        Returns:
            np.ndarray: Combined spike rates of shape (12000,)
        """
        # DÜZELTME: Return zeros if not initialized
        if not self.is_initialized:
            return np.zeros(12000)
            
        rates = self.get_spike_rates()
        
        # Ensure we return exactly 12000 values as expected by the AGI system
        target_size = 12000
        if len(rates) == target_size:
            return rates
        elif len(rates) < target_size:
            # Pad with zeros if too small
            padded = np.zeros(target_size)
            padded[:len(rates)] = rates
            return padded
        else:
            # Truncate if too large
            return rates[:target_size]
    
    def release(self):
        """Release all encoder resources."""
        if self.video_encoder:
            self.video_encoder.release()
        
        logger.info("RealtimeEncoder resources released")