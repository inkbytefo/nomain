## Developer: inkbytefo
## Modified: 2025-11-08

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from brian2 import SpikeGeneratorGroup, ms, Hz

logger = logging.getLogger(__name__)


class TextToConcepts:
    """
    Converts text input to neural spike patterns for concept representation.
    Maintains a vocabulary and triggers specific spike patterns for known words.
    """
    
    def __init__(self, vocab_size: int = 100, concept_neurons: int = 50):
        """
        Initialize the text-to-concepts converter.
        
        Args:
            vocab_size: Maximum number of words in vocabulary
            concept_neurons: Number of neurons per concept representation
        """
        self.vocab_size = vocab_size
        self.concept_neurons = concept_neurons
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.concept_patterns: Dict[int, np.ndarray] = {}
        self.next_word_id = 0
        
        # Initialize with basic vocabulary
        self._initialize_basic_vocabulary()
        
        # Brian2 objects
        self.spike_generator: Optional[SpikeGeneratorGroup] = None
        self.current_spikes: List[Tuple[float, int]] = []
        
    def _initialize_basic_vocabulary(self):
        """Initialize with a basic set of words for object and property recognition."""
        basic_words = [
            # Objects
            'cup', 'ball', 'book', 'phone', 'key', 'pen', 'paper', 'computer',
            'table', 'chair', 'door', 'window', 'light', 'bottle', 'glass',
            # Properties
            'big', 'small', 'red', 'blue', 'green', 'white', 'black', 'yellow',
            'hot', 'cold', 'heavy', 'light', 'soft', 'hard', 'smooth', 'rough',
            # Actions
            'see', 'hear', 'touch', 'move', 'stop', 'go', 'take', 'put',
            # Questions
            'what', 'where', 'when', 'why', 'how', 'who', 'which',
            # Connectives
            'the', 'a', 'an', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at'
        ]
        
        for word in basic_words:
            self.add_word(word)
    
    def add_word(self, word: str) -> int:
        """
        Add a new word to the vocabulary.
        
        Args:
            word: The word to add
            
        Returns:
            The assigned word ID
        """
        word_lower = word.lower()
        if word_lower in self.word_to_id:
            return self.word_to_id[word_lower]
        
        if self.next_word_id >= self.vocab_size:
            logger.warning(f"Vocabulary full ({self.vocab_size} words). Cannot add '{word}'")
            return -1
        
        word_id = self.next_word_id
        self.word_to_id[word_lower] = word_id
        self.id_to_word[word_id] = word_lower
        
        # Generate a unique sparse spike pattern for this concept
        # Each concept is represented by a specific pattern of active neurons
        pattern = self._generate_concept_pattern(word_id)
        self.concept_patterns[word_id] = pattern
        
        self.next_word_id += 1
        logger.debug(f"Added word '{word}' with ID {word_id}")
        
        return word_id
    
    def _generate_concept_pattern(self, word_id: int) -> np.ndarray:
        """
        Generate a unique sparse spike pattern for a concept.
        
        Args:
            word_id: The word ID to generate pattern for
            
        Returns:
            Binary array indicating which neurons should fire
        """
        # Create a deterministic but unique pattern using hash
        np.random.seed(word_id * 42)  # Deterministic seed
        
        # Each concept activates ~20% of concept neurons
        num_active = max(3, self.concept_neurons // 5)
        pattern = np.zeros(self.concept_neurons, dtype=int)
        
        # Select random neurons to be active
        active_indices = np.random.choice(self.concept_neurons, num_active, replace=False)
        pattern[active_indices] = 1
        
        return pattern
    
    def process_text(self, text: str, current_time: float = 0.0) -> Optional[SpikeGeneratorGroup]:
        """
        Process input text and generate spike patterns.
        
        Args:
            text: Input text to process
            current_time: Current simulation time in seconds
            
        Returns:
            SpikeGeneratorGroup with the word spike patterns
        """
        words = self._tokenize(text)
        if not words:
            return None
        
        # Collect spike times and neuron indices
        spike_times = []
        spike_indices = []
        
        for i, word in enumerate(words):
            word_id = self.word_to_id.get(word.lower())
            if word_id is not None:
                pattern = self.concept_patterns[word_id]
                
                # Add spikes for this word pattern
                # Stagger words slightly in time for sequence processing
                word_time = current_time + i * 0.01  # 10ms between words
                
                for neuron_idx in np.where(pattern == 1)[0]:
                    # Map to global neuron index
                    global_idx = word_id * self.concept_neurons + neuron_idx
                    spike_times.append(word_time)
                    spike_indices.append(global_idx)
        
        if not spike_times:
            return None
        
        # Create spike generator
        spike_times = np.array(spike_times) / ms  # Convert to Brian2 time units
        spike_indices = np.array(spike_indices)
        
        total_neurons = self.vocab_size * self.concept_neurons
        self.spike_generator = SpikeGeneratorGroup(total_neurons, spike_indices, spike_times)
        
        logger.info(f"Processed text '{text}': {len(words)} words, {len(spike_times)} spikes")
        return self.spike_generator
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization - split on whitespace and remove punctuation.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Remove common punctuation and split
        import re
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [token for token in cleaned.split() if token]
        return tokens
    
    def get_concept_neurons_count(self) -> int:
        """Get total number of concept neurons."""
        return self.vocab_size * self.concept_neurons
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        return {
            'vocab_size': self.vocab_size,
            'words_learned': len(self.word_to_id),
            'concept_neurons_per_word': self.concept_neurons,
            'total_concept_neurons': self.get_concept_neurons_count()
        }
    
    def word_exists(self, word: str) -> bool:
        """Check if a word exists in the vocabulary."""
        return word.lower() in self.word_to_id
    
    def get_word_id(self, word: str) -> Optional[int]:
        """Get the ID for a word, or None if not found."""
        return self.word_to_id.get(word.lower())
    
    def get_word_pattern(self, word: str) -> Optional[np.ndarray]:
        """Get the spike pattern for a word, or None if not found."""
        word_id = self.get_word_id(word)
        if word_id is not None:
            return self.concept_patterns[word_id]
        return None
    
    def learn_new_word(self, word: str) -> bool:
        """
        Learn a new word and add it to vocabulary.
        
        Args:
            word: New word to learn
            
        Returns:
            True if word was added, False if vocabulary full or already exists
        """
        if self.word_exists(word):
            logger.debug(f"Word '{word}' already exists in vocabulary")
            return False
        
        word_id = self.add_word(word)
        return word_id != -1
    
    def get_active_concepts(self, spike_indices: np.ndarray, spike_times: np.ndarray, 
                          time_window: float = 0.1) -> Dict[str, float]:
        """
        Determine which concepts are currently active based on recent spikes.
        
        Args:
            spike_indices: Array of spike neuron indices
            spike_times: Array of spike times (in seconds)
            time_window: Time window to consider for "current" activity (seconds)
            
        Returns:
            Dictionary mapping words to activation levels
        """
        current_time = np.max(spike_times) if len(spike_times) > 0 else 0.0
        recent_mask = (current_time - spike_times) <= time_window
        
        if not np.any(recent_mask):
            return {}
        
        recent_spikes = spike_indices[recent_mask]
        concept_activations = {}
        
        # Count spikes per concept
        for spike_idx in recent_spikes:
            word_id = spike_idx // self.concept_neurons
            if word_id in self.id_to_word:
                word = self.id_to_word[word_id]
                concept_activations[word] = concept_activations.get(word, 0) + 1
        
        # Normalize by maximum possible spikes per concept
        for word in concept_activations:
            concept_activations[word] /= self.concept_neurons
        
        return concept_activations