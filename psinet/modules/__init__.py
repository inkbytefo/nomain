## Developer: inkbytefo
## Modified: 2025-11-09

"""
Chimera AGI Modules

This package contains the 5 core modules of the Chimera AGI architecture:

1. Sensory Cortex - Vision and audio processing with feature extraction
2. World Model - Concept formation and memory (Hippocampus)
3. Language Module - Text-concept interface with hybrid LLM approach
4. Motivation System - Curiosity-driven learning and goal orientation
"""

from .sensory_cortex import SensoryCortex
from .world_model import WorldModel
from .language_module import LanguageModule
from .motivation_system import MotivationSystem

__all__ = [
    'SensoryCortex',
    'WorldModel', 
    'LanguageModule',
    'MotivationSystem'
]