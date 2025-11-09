## Developer: inkbytefo
## Modified: 2025-11-09

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class WorldModel:
    """
    Module 3: World Model & Hippocampus - Concept Formation and Memory
    
    Creates stable activation patterns representing concepts and stores/retrieves
    them through hippocampal memory mechanisms. This is the core of the AGI's
    "understanding" - where sensory patterns become meaningful concepts.
    """
    
    def __init__(self, associative_cortex, hippocampus, config: Dict[str, Any]):
        """
        Initialize World Model with neural components.
        
        Args:
            associative_cortex: Associative cortex neuron population
            hippocampus: Hippocampus neuron population  
            config: World model configuration
        """
        self.associative_cortex = associative_cortex
        self.hippocampus = hippocampus
        self.config = config
        
        # Concept formation parameters
        self.concept_threshold = config.get('concept_formation_threshold', 0.5)
        self.pattern_stability_time = config.get('pattern_stability_time', 2.0)
        self.prediction_error_window = config.get('prediction_error_window', 1.0)
        self.memory_consolidation_interval = config.get('memory_consolidation_interval', 10.0)
        
        # Concept storage
        self.concepts = {}  # name -> ConceptPattern
        self.active_concepts = {}  # concept_id -> activation_strength
        self.concept_history = deque(maxlen=1000)
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(capacity=1000)
        self.semantic_memory = SemanticMemory()
        self.working_memory = WorkingMemory(capacity=50)
        
        # Prediction and learning
        self.predictor = PatternPredictor(
            input_size=associative_cortex.num_neurons,
            hidden_size=hippocampus.num_neurons
        )
        
        # Timing
        self.last_consolidation_time = 0.0
        self.simulation_time = 0.0
        
        logger.info("World Model & Hippocampus initialized")
    
    def update(self, neurons: Dict[str, Any], simulation_time: float):
        """
        Update world model with current neural state.
        
        Args:
            neurons: Dictionary of neuron populations
            simulation_time: Current simulation time
        """
        self.simulation_time = simulation_time
        
        # Get current associative cortex activity
        if 'associative_cortex' in neurons:
            current_pattern = neurons['associative_cortex'].get_membrane_potentials()
            current_spikes = neurons['associative_cortex'].get_spikes()
            
            # Update working memory with current pattern
            self.working_memory.add_pattern(current_pattern, simulation_time)
            
            # Detect and form concepts
            self._detect_concepts(current_pattern, current_spikes, simulation_time)
            
            # Update predictions
            self._update_predictions(current_pattern, simulation_time)
            
            # Consolidate memories periodically
            if simulation_time - self.last_consolidation_time > self.memory_consolidation_interval:
                self._consolidate_memories()
                self.last_consolidation_time = simulation_time
    
    def _detect_concepts(self, pattern: np.ndarray, spikes: np.ndarray, simulation_time: float):
        """
        Detect stable patterns and form concepts.
        
        Args:
            pattern: Current activation pattern
            spikes: Current spike pattern
            simulation_time: Current time
        """
        # Check for stable patterns in working memory
        stable_patterns = self.working_memory.get_stable_patterns(
            stability_threshold=self.concept_threshold,
            min_duration=self.pattern_stability_time,
            current_time=simulation_time
        )
        
        for stable_pattern in stable_patterns:
            # Check if this matches existing concepts
            concept_id = self._match_existing_concept(stable_pattern['pattern'])
            
            if concept_id is None:
                # Create new concept
                concept_id = self._create_new_concept(stable_pattern['pattern'], simulation_time)
                logger.info(f"New concept formed: {concept_id}")
            
            # Update concept activation
            self.active_concepts[concept_id] = stable_pattern['stability']
            
            # Store in episodic memory
            self.episodic_memory.add_episode(
                pattern=stable_pattern['pattern'],
                concept_id=concept_id,
                timestamp=simulation_time
            )
    
    def _match_existing_concept(self, pattern: np.ndarray) -> Optional[str]:
        """
        Match pattern against existing concepts.
        
        Args:
            pattern: Activation pattern to match
            
        Returns:
            Concept ID if match found, None otherwise
        """
        best_match = None
        best_similarity = 0.0
        
        for concept_id, concept in self.concepts.items():
            similarity = concept.calculate_similarity(pattern)
            if similarity > best_similarity and similarity > self.concept_threshold:
                best_similarity = similarity
                best_match = concept_id
        
        return best_match
    
    def _create_new_concept(self, pattern: np.ndarray, simulation_time: float) -> str:
        """
        Create a new concept from stable pattern.
        
        Args:
            pattern: Stable activation pattern
            simulation_time: Creation time
            
        Returns:
            New concept ID
        """
        concept_id = f"concept_{len(self.concepts)}_{int(simulation_time)}"
        
        # Create concept pattern
        concept = ConceptPattern(
            pattern=pattern.copy(),
            creation_time=simulation_time,
            concept_id=concept_id
        )
        
        # Store concept
        self.concepts[concept_id] = concept
        
        # Add to semantic memory
        self.semantic_memory.add_concept(concept)
        
        # Record concept formation
        self.concept_history.append({
            'concept_id': concept_id,
            'timestamp': simulation_time,
            'pattern': pattern.copy()
        })
        
        return concept_id
    
    def _update_predictions(self, current_pattern: np.ndarray, simulation_time: float):
        """
        Update pattern predictions and calculate prediction error.
        
        Args:
            current_pattern: Current activation pattern
            simulation_time: Current time
        """
        # Get recent patterns for prediction context
        recent_patterns = self.working_memory.get_recent_patterns(
            window_size=self.prediction_error_window,
            current_time=simulation_time
        )
        
        if len(recent_patterns) >= 2:
            # Predict next pattern
            predicted_pattern = self.predictor.predict(recent_patterns[:-1])
            
            # Calculate prediction error
            prediction_error = np.mean(np.abs(predicted_pattern - current_pattern))
            
            # Update predictor with actual pattern
            self.predictor.update(recent_patterns, current_pattern)
            
            # Store prediction error for motivation system
            self.last_prediction_error = prediction_error
    
    def _consolidate_memories(self):
        """Consolidate memories from episodic to semantic memory."""
        # Get important episodes for consolidation
        important_episodes = self.episodic_memory.get_important_episodes(
            importance_threshold=0.7
        )
        
        for episode in important_episodes:
            # Strengthen semantic connections
            self.semantic_memory.strengthen_connections(
                concept_id=episode['concept_id'],
                strength=episode['importance']
            )
        
        logger.info(f"Consolidated {len(important_episodes)} memories")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current world model state."""
        return {
            'active_concepts': self.active_concepts.copy(),
            'concept_count': len(self.concepts),
            'episodic_memory_size': len(self.episodic_memory),
            'working_memory_load': len(self.working_memory),
            'prediction_error': getattr(self, 'last_prediction_error', 0.0)
        }
    
    def recall_concept(self, concept_id: str) -> Optional[np.ndarray]:
        """
        Recall a concept pattern.
        
        Args:
            concept_id: ID of concept to recall
            
        Returns:
            Concept pattern if found, None otherwise
        """
        if concept_id in self.concepts:
            return self.concepts[concept_id].pattern.copy()
        return None
    
    def get_similar_concepts(self, pattern: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get concepts most similar to given pattern.
        
        Args:
            pattern: Query pattern
            top_k: Number of similar concepts to return
            
        Returns:
            List of (concept_id, similarity) tuples
        """
        similarities = []
        for concept_id, concept in self.concepts.items():
            similarity = concept.calculate_similarity(pattern)
            similarities.append((concept_id, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def reset(self):
        """Reset world model state."""
        self.active_concepts.clear()
        self.concept_history.clear()
        self.episodic_memory.clear()
        self.working_memory.clear()
        self.predictor.reset()
        self.last_consolidation_time = 0.0
        self.simulation_time = 0.0
        logger.info("World Model reset")


class ConceptPattern:
    """Represents a stable concept pattern."""
    
    def __init__(self, pattern: np.ndarray, creation_time: float, concept_id: str):
        """
        Initialize concept pattern.
        
        Args:
            pattern: Activation pattern representing the concept
            creation_time: When the concept was formed
            concept_id: Unique identifier
        """
        self.pattern = pattern
        self.creation_time = creation_time
        self.concept_id = concept_id
        self.activation_count = 0
        self.last_activation = creation_time
        self.strength = 1.0
    
    def activate(self, activation_time: float):
        """Record concept activation."""
        self.activation_count += 1
        self.last_activation = activation_time
        # Strengthen concept with repeated activation
        self.strength = min(2.0, self.strength * 1.01)
    
    def calculate_similarity(self, other_pattern: np.ndarray) -> float:
        """
        Calculate similarity with another pattern.
        
        Args:
            other_pattern: Pattern to compare with
            
        Returns:
            Similarity score (0-1)
        """
        # Use cosine similarity
        dot_product = np.dot(self.pattern, other_pattern)
        norm_a = np.linalg.norm(self.pattern)
        norm_b = np.linalg.norm(other_pattern)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, similarity)


class EpisodicMemory:
    """Episodic memory system for storing experiences."""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize episodic memory.
        
        Args:
            capacity: Maximum number of episodes to store
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
    
    def add_episode(self, pattern: np.ndarray, concept_id: str, timestamp: float):
        """
        Add a new episode.
        
        Args:
            pattern: Activation pattern
            concept_id: Associated concept
            timestamp: Episode time
        """
        episode = {
            'pattern': pattern.copy(),
            'concept_id': concept_id,
            'timestamp': timestamp,
            'importance': self._calculate_importance(pattern)
        }
        self.episodes.append(episode)
    
    def _calculate_importance(self, pattern: np.ndarray) -> float:
        """Calculate importance of an episode based on pattern activation."""
        # Simple importance based on activation strength
        return np.mean(np.abs(pattern))
    
    def get_important_episodes(self, importance_threshold: float) -> List[Dict[str, Any]]:
        """
        Get episodes above importance threshold.
        
        Args:
            importance_threshold: Minimum importance score
            
        Returns:
            List of important episodes
        """
        return [ep for ep in self.episodes if ep['importance'] > importance_threshold]
    
    def clear(self):
        """Clear episodic memory."""
        self.episodes.clear()
    
    def __len__(self):
        return len(self.episodes)


class SemanticMemory:
    """Semantic memory system for long-term concept storage."""
    
    def __init__(self):
        """Initialize semantic memory."""
        self.concepts = {}
        self.connections = {}  # concept_id -> {other_concept_id: strength}
    
    def add_concept(self, concept: ConceptPattern):
        """
        Add a concept to semantic memory.
        
        Args:
            concept: Concept to add
        """
        self.concepts[concept.concept_id] = concept
        if concept.concept_id not in self.connections:
            self.connections[concept.concept_id] = {}
    
    def strengthen_connections(self, concept_id: str, strength: float):
        """
        Strengthen connections for a concept.
        
        Args:
            concept_id: Concept to strengthen
            strength: Connection strength to add
        """
        if concept_id in self.connections:
            for other_concept in self.connections[concept_id]:
                self.connections[concept_id][other_concept] += strength


class WorkingMemory:
    """Working memory for temporary pattern storage."""
    
    def __init__(self, capacity: int = 50):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of patterns to store
        """
        self.capacity = capacity
        self.patterns = deque(maxlen=capacity)
    
    def add_pattern(self, pattern: np.ndarray, timestamp: float):
        """
        Add a pattern to working memory.
        
        Args:
            pattern: Activation pattern
            timestamp: Pattern time
        """
        self.patterns.append({
            'pattern': pattern.copy(),
            'timestamp': timestamp
        })
    
    def get_stable_patterns(self, stability_threshold: float, min_duration: float, 
                          current_time: float) -> List[Dict[str, Any]]:
        """
        Get patterns that have been stable over time.
        
        Args:
            stability_threshold: Minimum stability score
            min_duration: Minimum duration for stability
            current_time: Current simulation time
            
        Returns:
            List of stable patterns
        """
        stable_patterns = []
        
        # Group patterns by similarity
        pattern_groups = self._group_similar_patterns()
        
        for group in pattern_groups:
            if len(group) >= 2:
                # Check temporal stability
                time_span = group[-1]['timestamp'] - group[0]['timestamp']
                if time_span >= min_duration:
                    # Calculate pattern stability
                    avg_pattern = np.mean([p['pattern'] for p in group], axis=0)
                    variance = np.var([p['pattern'] for p in group], axis=0)
                    stability = 1.0 - np.mean(variance)
                    
                    if stability > stability_threshold:
                        stable_patterns.append({
                            'pattern': avg_pattern,
                            'stability': stability,
                            'duration': time_span
                        })
        
        return stable_patterns
    
    def get_recent_patterns(self, window_size: float, current_time: float) -> List[np.ndarray]:
        """
        Get patterns within time window.
        
        Args:
            window_size: Time window size
            current_time: Current time
            
        Returns:
            List of recent patterns
        """
        cutoff_time = current_time - window_size
        recent = [p['pattern'] for p in self.patterns if p['timestamp'] >= cutoff_time]
        return recent
    
    def _group_similar_patterns(self, similarity_threshold: float = 0.8) -> List[List[Dict[str, Any]]]:
        """Group similar patterns together."""
        if len(self.patterns) == 0:
            return []
        
        groups = []
        ungrouped = list(self.patterns)
        
        while ungrouped:
            # Start new group with first ungrouped pattern
            current_group = [ungrouped.pop(0)]
            
            # Find similar patterns
            i = 0
            while i < len(ungrouped):
                pattern = ungrouped[i]
                # Check similarity with group representative
                similarity = self._calculate_similarity(
                    current_group[0]['pattern'], pattern['pattern']
                )
                
                if similarity > similarity_threshold:
                    current_group.append(ungrouped.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def clear(self):
        """Clear working memory."""
        self.patterns.clear()


class PatternPredictor:
    """Simple pattern prediction system."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize pattern predictor.
        
        Args:
            input_size: Size of input patterns
            hidden_size: Size of hidden representation
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Simple linear predictor (could be replaced with more sophisticated model)
        self.weights = np.random.randn(hidden_size, input_size) * 0.1
        self.output_weights = np.random.randn(input_size, hidden_size) * 0.1
        
        self.learning_rate = 0.01
    
    def predict(self, pattern_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Predict next pattern in sequence.
        
        Args:
            pattern_sequence: Sequence of patterns
            
        Returns:
            Predicted next pattern
        """
        if not pattern_sequence:
            return np.zeros(self.input_size)
        
        # Use last pattern as input (simplified)
        last_pattern = pattern_sequence[-1]
        
        # Forward pass
        hidden = np.tanh(np.dot(self.weights, last_pattern))
        output = np.dot(self.output_weights, hidden)
        
        return output
    
    def update(self, pattern_sequence: List[np.ndarray], target_pattern: np.ndarray):
        """
        Update predictor with new example.
        
        Args:
            pattern_sequence: Input pattern sequence
            target_pattern: Target pattern to predict
        """
        if not pattern_sequence:
            return
        
        last_pattern = pattern_sequence[-1]
        
        # Forward pass
        hidden = np.tanh(np.dot(self.weights, last_pattern))
        output = np.dot(self.output_weights, hidden)
        
        # Calculate error
        error = target_pattern - output
        
        # Backward pass (simplified gradient descent)
        hidden_error = np.dot(self.output_weights.T, error)
        hidden_gradient = hidden_error * (1 - hidden ** 2)
        
        # Update weights
        self.output_weights += self.learning_rate * np.outer(error, hidden)
        self.weights += self.learning_rate * np.outer(hidden_gradient, last_pattern)
    
    def reset(self):
        """Reset predictor weights."""
        self.weights = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.output_weights = np.random.randn(self.input_size, self.hidden_size) * 0.1