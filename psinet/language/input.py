## Developer: inkbytefo
## Modified: 2025-11-08

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
# Brian2 bağımlılıkları kaldırıldı
# from brian2 import SpikeGeneratorGroup, ms, Hz

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
        
        self._initialize_basic_vocabulary()
        
    def _initialize_basic_vocabulary(self):
        """Initialize with a basic set of words for object and property recognition."""
        basic_words = [
            'cup', 'ball', 'book', 'phone', 'key', 'pen', 'paper', 'computer',
            'table', 'chair', 'door', 'window', 'light', 'bottle', 'glass',
            'big', 'small', 'red', 'blue', 'green', 'white', 'black', 'yellow',
            'hot', 'cold', 'heavy', 'light', 'soft', 'hard', 'smooth', 'rough',
            'see', 'hear', 'touch', 'move', 'stop', 'go', 'take', 'put',
            'what', 'where', 'when', 'why', 'how', 'who', 'which',
            'the', 'a', 'an', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at'
        ]
        
        for word in basic_words:
            self.add_word(word)
    
    def add_word(self, word: str) -> int:
        """
        Add a new word to the vocabulary.
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
        
        pattern = self._generate_concept_pattern(word_id)
        self.concept_patterns[word_id] = pattern
        
        self.next_word_id += 1
        return word_id
    
    def _generate_concept_pattern(self, word_id: int) -> np.ndarray:
        """
        Generate a unique sparse spike pattern for a concept.
        """
        np.random.seed(word_id * 42)
        num_active = max(3, self.concept_neurons // 5)
        pattern = np.zeros(self.concept_neurons, dtype=int)
        active_indices = np.random.choice(self.concept_neurons, num_active, replace=False)
        pattern[active_indices] = 1
        return pattern
    
    # DEĞİŞİKLİK: Bu metot artık Brian2 nesnesi yerine saf numpy dizileri döndürüyor.
    def text_to_spike_data(self, text: str, current_time: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process input text and generate spike data (indices and times).

        Args:
            text: Input text to process
            current_time: Current simulation time in seconds
            
        Returns:
            A tuple of (spike_indices, spike_times) or None.
        """
        words = self._tokenize(text)
        if not words:
            return None
        
        spike_times = []
        spike_indices = []
        
        for i, word in enumerate(words):
            word_id = self.word_to_id.get(word.lower())
            if word_id is not None:
                pattern = self.concept_patterns[word_id]
                word_time = current_time + i * 0.01
                
                for neuron_idx in np.where(pattern == 1)[0]:
                    # Bu, dil nöronlarının genel nöron popülasyonundaki yerini belirtir.
                    # 'core_thread' bu indeksleri kullanarak doğru nöronları aktive eder.
                    global_idx = neuron_idx # word_id * self.concept_neurons + neuron_idx
                    spike_times.append(word_time)
                    spike_indices.append(global_idx)
        
        if not spike_times:
            return None
        
        logger.info(f"Processed text '{text}': {len(words)} words, {len(spike_times)} spikes")
        return (np.array(spike_indices, dtype=np.int32), np.array(spike_times, dtype=np.float32))

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        """
        import re
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [token for token in cleaned.split() if token]
        return tokens
    
    def get_concept_neurons_count(self) -> int:
        return self.vocab_size * self.concept_neurons

    def get_vocabulary_stats(self) -> Dict[str, int]:
        return {
            'vocab_size': self.vocab_size,
            'words_learned': len(self.word_to_id),
            'concept_neurons_per_word': self.concept_neurons,
            'total_concept_neurons': self.get_concept_neurons_count()
        }