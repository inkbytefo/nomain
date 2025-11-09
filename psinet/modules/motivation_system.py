## Developer: inkbytefo
## Modified: 2025-11-09

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time
import math

logger = logging.getLogger(__name__)


class MotivationSystem:
    """
    Module 5: Motivation System - Curiosity and Goal Orientation
    
    Implements biologically-inspired motivation mechanisms:
    - Curiosity: Driven by prediction errors and novelty detection
    - Goal Orientation: Maintains persistent goals and reward signals
    - Dopamine Modulation: Global neuromodulation for learning regulation
    
    This system provides the "why" behind the AGI's behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Motivation System.
        
        Args:
            config: Motivation system configuration
        """
        self.config = config
        
        # Curiosity parameters
        curiosity_config = config.get('curiosity', {})
        self.curiosity_enabled = curiosity_config.get('enable', True)
        self.prediction_error_scale = curiosity_config.get('prediction_error_scale', 1.0)
        self.novelty_threshold = curiosity_config.get('novelty_threshold', 0.3)
        self.dopamine_decay = curiosity_config.get('dopamine_decay', 0.95)
        
        # Goal orientation parameters
        goal_config = config.get('goal_orientation', {})
        self.goal_orientation_enabled = goal_config.get('enable', True)
        self.goal_persistence = goal_config.get('goal_persistence', 5.0)
        self.reward_scale = goal_config.get('reward_scale', 1.0)
        self.punishment_scale = goal_config.get('punishment_scale', -0.5)
        
        # Internal state
        self.dopamine_level = 1.0  # Baseline dopamine
        self.current_goals = {}  # goal_id -> GoalState
        self.prediction_errors = deque(maxlen=100)
        self.novelty_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=100)
        
        # Curiosity components
        self.novelty_detector = NoveltyDetector(threshold=self.novelty_threshold)
        self.prediction_error_calculator = PredictionErrorCalculator()
        
        # Goal management
        self.goal_manager = GoalManager(persistence_time=self.goal_persistence)
        
        # Motivation signals for different brain regions
        self.region_modulation = {
            'vision_to_assoc': 1.0,
            'audio_to_assoc': 1.0,
            'assoc_to_hippocampus': 1.0,
            'hippocampus_to_assoc': 1.0,
            'assoc_to_language': 1.0,
            'language_to_assoc': 1.0,
            'assoc_to_motor': 1.0
        }
        
        # Prediction error tracking for dopamine formulation
        self.baseline_error = 0.5  # Baseline error level
        self.error_history = deque(maxlen=100)
        self.k_gain = 2.0  # Gain factor for error-to-dopamine conversion
        
        # Timing
        self.last_update_time = 0.0
        self.simulation_time = 0.0
        
        logger.info("Motivation System initialized")
    
    def update(self, neurons: Dict[str, Any], world_model, simulation_time: float) -> Dict[str, float]:
        """
        Update motivation system and generate neuromodulation signals.
        
        Args:
            neurons: Dictionary of neuron populations
            world_model: World model instance
            simulation_time: Current simulation time
            
        Returns:
            Dictionary of dopamine modulation signals for synapses
        """
        self.simulation_time = simulation_time
        dt = simulation_time - self.last_update_time
        self.last_update_time = simulation_time
        
        # Calculate prediction error for dopamine formulation
        prediction_error = self._calculate_prediction_error(neurons, world_model)
        
        # Update curiosity
        if self.curiosity_enabled:
            curiosity_signal = self._update_curiosity(neurons, world_model, prediction_error)
        else:
            curiosity_signal = 0.0
        
        # Update goal orientation
        if self.goal_orientation_enabled:
            goal_signal = self._update_goal_orientation(neurons, world_model)
        else:
            goal_signal = 0.0
        
        # Combine motivation signals
        total_motivation = curiosity_signal + goal_signal
        
        # Update dopamine level with precise formulation
        self._update_dopamine_precise(prediction_error, total_motivation, dt)
        
        # Generate region-specific modulation
        modulation_signals = self._generate_modulation_signals(curiosity_signal, goal_signal)
        
        return modulation_signals
    
    def _calculate_prediction_error(self, neurons: Dict[str, Any], world_model) -> float:
        """
        Calculate prediction error using precise mathematical formulation.
        
        Formula: PredictiveError = ||ActualSensoryInput - PredictedSensoryInput||
        
        Args:
            neurons: Current neural state
            world_model: World model instance
            
        Returns:
            Prediction error value (0-1)
        """
        prediction_error = 0.0
        
        # Get actual sensory input
        if 'associative_cortex' in neurons:
            actual_pattern = neurons['associative_cortex'].get_membrane_potentials()
            
            # Get predicted pattern from world model
            if hasattr(world_model, 'predictor') and world_model.predictor:
                # Get recent patterns for prediction context
                recent_patterns = getattr(world_model, 'working_memory', None)
                if recent_patterns and hasattr(recent_patterns, 'patterns') and len(recent_patterns.patterns) > 0:
                    # Use last pattern as prediction context
                    last_pattern = recent_patterns.patterns[-1]
                    predicted_pattern = world_model.predictor.predict(last_pattern)
                    
                    # Calculate Euclidean distance between actual and predicted
                    error_vector = actual_pattern - predicted_pattern
                    prediction_error = np.linalg.norm(error_vector) / np.linalg.norm(actual_pattern)
                    
                    # Normalize to 0-1 range
                    prediction_error = min(1.0, prediction_error)
                else:
                    # No prediction available, use baseline
                    prediction_error = self.baseline_error
            else:
                # No predictor available, use baseline
                prediction_error = self.baseline_error
        
        # Update error history
        self.error_history.append(prediction_error)
        
        # Update baseline error (exponential moving average)
        if len(self.error_history) > 10:
            self.baseline_error = 0.9 * self.baseline_error + 0.1 * np.mean(list(self.error_history)[-10:])
        
        return prediction_error
    
    def _update_dopamine_precise(self, prediction_error: float, motivation_signal: float, dt: float):
        """
        Update dopamine level with precise mathematical formulation.
        
        Formula: DopamineLevel = 1.0 + clip(k * (PredictiveError - BaselineError), -0.5, 1.0)
        
        Args:
            prediction_error: Current prediction error
            motivation_signal: Additional motivation signal
            dt: Time step
        """
        # Calculate dopamine from prediction error
        error_dopamine = self.k_gain * (prediction_error - self.baseline_error)
        
        # Clip to reasonable range
        error_dopamine = np.clip(error_dopamine, -0.5, 1.0)
        
        # Combine with baseline and motivation signal
        target_dopamine = 1.0 + error_dopamine + motivation_signal * 0.5
        
        # Smooth dopamine changes with time constant
        tau_dopamine = 0.5  # Dopamine time constant in seconds
        alpha = dt / tau_dopamine
        self.dopamine_level += alpha * (target_dopamine - self.dopamine_level)
        
        # Apply decay
        self.dopamine_level *= self.dopamine_decay
        
        # Clamp to physiological range
        self.dopamine_level = np.clip(self.dopamine_level, 0.1, 3.0)
    
    def _update_curiosity(self, neurons: Dict[str, Any], world_model, prediction_error: float) -> float:
        """
        Update curiosity based on prediction errors and novelty.
        
        Args:
            neurons: Neuron populations
            world_model: World model instance
            
        Returns:
            Curiosity signal strength
        """
        curiosity_signal = 0.0
        
        # Use the precise prediction error calculation
        curiosity_signal += prediction_error * self.prediction_error_scale
        
        # Calculate novelty from neural activity
        if 'associative_cortex' in neurons:
            current_activity = neurons['associative_cortex'].get_membrane_potentials()
            novelty_score = self.novelty_detector.calculate_novelty(current_activity)
            self.novelty_history.append(novelty_score)
            
            # Add novelty to curiosity signal
            if novelty_score > self.novelty_threshold:
                curiosity_signal += novelty_score * 2.0
        
        # Bonus for very high prediction errors (surprise)
        if len(self.prediction_errors) > 0:
            recent_errors = list(self.prediction_errors)[-10:]
            max_error = max(recent_errors)
            if max_error > 0.8:
                curiosity_signal += 1.0  # Strong surprise bonus
        
        return max(0.0, curiosity_signal)
    
    def _update_goal_orientation(self, neurons: Dict[str, Any], world_model) -> float:
        """
        Update goal orientation based on current goals and progress.
        
        Args:
            neurons: Neuron populations
            world_model: World model instance
            
        Returns:
            Goal signal strength
        """
        goal_signal = 0.0
        
        # Update existing goals
        self.goal_manager.update_goals(neurons, world_model, self.simulation_time)
        
        # Calculate goal progress
        goal_progress = self.goal_manager.calculate_goal_progress(neurons, world_model)
        
        # Generate reward/punishment based on goal progress
        for goal_id, progress in goal_progress.items():
            if progress > 0.8:  # Goal achieved
                reward = self.reward_scale
                self.reward_history.append(reward)
                goal_signal += reward
                logger.info(f"Goal achieved: {goal_id}")
            elif progress < 0.2:  # Goal failing
                punishment = self.punishment_scale
                self.reward_history.append(punishment)
                goal_signal += punishment
        
        # Maintain baseline goal motivation
        if len(self.goal_manager.active_goals) > 0:
            goal_signal += 0.1  # Small baseline motivation for having goals
        
        return goal_signal
    
    def _update_dopamine(self, motivation_signal: float, dt: float):
        """
        Update global dopamine level.
        
        Args:
            motivation_signal: Current motivation signal
            dt: Time step
        """
        # Dopamine responds to motivation signals
        target_dopamine = 1.0 + motivation_signal
        
        # Smooth dopamine changes
        self.dopamine_level += (target_dopamine - self.dopamine_level) * dt * 2.0
        
        # Apply decay
        self.dopamine_level *= self.dopamine_decay
        
        # Clamp to reasonable range
        self.dopamine_level = np.clip(self.dopamine_level, 0.1, 3.0)
    
    def _generate_modulation_signals(self, curiosity_signal: float, goal_signal: float) -> Dict[str, float]:
        """
        Generate region-specific dopamine modulation signals.
        
        Args:
            curiosity_signal: Current curiosity signal
            goal_signal: Current goal signal
            
        Returns:
            Dictionary of modulation signals for each synapse type
        """
        modulation = self.region_modulation.copy()
        
        # Base modulation from global dopamine
        base_modulation = self.dopamine_level
        
        # Curiosity enhances learning in sensory pathways
        if curiosity_signal > 0.5:
            modulation['vision_to_assoc'] *= (1.0 + curiosity_signal * 0.5)
            modulation['audio_to_assoc'] *= (1.0 + curiosity_signal * 0.5)
            modulation['assoc_to_hippocampus'] *= (1.0 + curiosity_signal * 0.3)
        
        # Goal orientation enhances motor and language pathways
        if goal_signal > 0.3:
            modulation['assoc_to_motor'] *= (1.0 + goal_signal * 0.4)
            modulation['assoc_to_language'] *= (1.0 + goal_signal * 0.3)
            modulation['language_to_assoc'] *= (1.0 + goal_signal * 0.2)
        
        # Apply global dopamine modulation
        for key in modulation:
            modulation[key] *= base_modulation
        
        return modulation
    
    def add_goal(self, goal_description: str, target_pattern: Optional[np.ndarray] = None, 
                priority: float = 1.0):
        """
        Add a new goal for the AGI.
        
        Args:
            goal_description: Text description of the goal
            target_pattern: Target neural pattern (optional)
            priority: Goal priority (0-1)
        """
        goal_id = f"goal_{len(self.goal_manager.active_goals)}_{int(self.simulation_time)}"
        
        goal = GoalState(
            goal_id=goal_id,
            description=goal_description,
            target_pattern=target_pattern,
            priority=priority,
            creation_time=self.simulation_time
        )
        
        self.goal_manager.add_goal(goal)
        logger.info(f"Goal added: {goal_description}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current motivation system state."""
        return {
            'dopamine_level': self.dopamine_level,
            'active_goals': len(self.goal_manager.active_goals),
            'recent_prediction_error': list(self.prediction_errors)[-1] if self.prediction_errors else 0.0,
            'recent_novelty': list(self.novelty_history)[-1] if self.novelty_history else 0.0,
            'recent_reward': list(self.reward_history)[-1] if self.reward_history else 0.0,
            'curiosity_enabled': self.curiosity_enabled,
            'goal_orientation_enabled': self.goal_orientation_enabled
        }
    
    def reset(self):
        """Reset motivation system state."""
        self.dopamine_level = 1.0
        self.current_goals.clear()
        self.prediction_errors.clear()
        self.novelty_history.clear()
        self.reward_history.clear()
        self.goal_manager.reset()
        self.novelty_detector.reset()
        self.last_update_time = 0.0
        self.simulation_time = 0.0
        logger.info("Motivation System reset")


class NoveltyDetector:
    """Detects novelty in neural activity patterns."""
    
    def __init__(self, threshold: float = 0.3, history_size: int = 20):
        """
        Initialize novelty detector.
        
        Args:
            threshold: Novelty detection threshold
            history_size: Number of past patterns to remember
        """
        self.threshold = threshold
        self.history_size = history_size
        self.pattern_history = deque(maxlen=history_size)
    
    def calculate_novelty(self, current_pattern: np.ndarray) -> float:
        """
        Calculate novelty score for current pattern.
        
        Args:
            current_pattern: Current neural activity pattern
            
        Returns:
            Novelty score (0-1)
        """
        if len(self.pattern_history) == 0:
            # First pattern is completely novel
            self.pattern_history.append(current_pattern.copy())
            return 1.0
        
        # Calculate similarity to past patterns
        min_similarity = 1.0
        
        for past_pattern in self.pattern_history:
            similarity = self._calculate_similarity(current_pattern, past_pattern)
            min_similarity = min(min_similarity, similarity)
        
        # Novelty is inverse of maximum similarity
        novelty = 1.0 - min_similarity
        
        # Add current pattern to history
        self.pattern_history.append(current_pattern.copy())
        
        return novelty
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        # Use cosine similarity
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)
    
    def reset(self):
        """Reset novelty detector."""
        self.pattern_history.clear()


class PredictionErrorCalculator:
    """Calculates prediction errors for curiosity signals."""
    
    def __init__(self):
        """Initialize prediction error calculator."""
        self.predictions = deque(maxlen=10)
        self.actuals = deque(maxlen=10)
    
    def calculate_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate prediction error.
        
        Args:
            predicted: Predicted pattern
            actual: Actual pattern
            
        Returns:
            Prediction error (0-1)
        """
        # Store for future reference
        self.predictions.append(predicted.copy())
        self.actuals.append(actual.copy())
        
        # Calculate mean squared error
        mse = np.mean((predicted - actual) ** 2)
        
        # Normalize to 0-1 range (approximate)
        normalized_error = min(1.0, mse / 10.0)
        
        return normalized_error
    
    def reset(self):
        """Reset prediction error calculator."""
        self.predictions.clear()
        self.actuals.clear()


class GoalManager:
    """Manages active goals and their progress."""
    
    def __init__(self, persistence_time: float = 5.0):
        """
        Initialize goal manager.
        
        Args:
            persistence_time: How long goals remain active (seconds)
        """
        self.persistence_time = persistence_time
        self.active_goals = {}  # goal_id -> GoalState
        self.goal_history = []
    
    def add_goal(self, goal: 'GoalState'):
        """
        Add a new goal.
        
        Args:
            goal: Goal to add
        """
        self.active_goals[goal.goal_id] = goal
    
    def update_goals(self, neurons: Dict[str, Any], world_model, current_time: float):
        """
        Update all active goals.
        
        Args:
            neurons: Current neural state
            world_model: World model instance
            current_time: Current simulation time
        """
        # Remove expired goals
        expired_goals = []
        for goal_id, goal in self.active_goals.items():
            if current_time - goal.creation_time > self.persistence_time:
                if goal.progress < 0.8:  # Goal not achieved
                    expired_goals.append(goal_id)
        
        for goal_id in expired_goals:
            del self.active_goals[goal_id]
            logger.info(f"Goal expired: {goal_id}")
        
        # Update goal progress
        for goal in self.active_goals.values():
            goal.update_progress(neurons, world_model, current_time)
    
    def calculate_goal_progress(self, neurons: Dict[str, Any], world_model) -> Dict[str, float]:
        """
        Calculate progress for all active goals.
        
        Args:
            neurons: Current neural state
            world_model: World model instance
            
        Returns:
            Dictionary mapping goal_id to progress (0-1)
        """
        progress = {}
        for goal_id, goal in self.active_goals.items():
            progress[goal_id] = goal.progress
        return progress
    
    def reset(self):
        """Reset goal manager."""
        self.active_goals.clear()
        self.goal_history.clear()


class GoalState:
    """Represents an active goal."""
    
    def __init__(self, goal_id: str, description: str, target_pattern: Optional[np.ndarray] = None,
                 priority: float = 1.0, creation_time: float = 0.0):
        """
        Initialize goal state.
        
        Args:
            goal_id: Unique goal identifier
            description: Goal description
            target_pattern: Target neural pattern
            priority: Goal priority
            creation_time: When goal was created
        """
        self.goal_id = goal_id
        self.description = description
        self.target_pattern = target_pattern
        self.priority = priority
        self.creation_time = creation_time
        self.progress = 0.0
        self.last_progress_update = creation_time
    
    def update_progress(self, neurons: Dict[str, Any], world_model, current_time: float):
        """
        Update goal progress based on current state.
        
        Args:
            neurons: Current neural state
            world_model: World model instance
            current_time: Current time
        """
        if self.target_pattern is not None and 'associative_cortex' in neurons:
            current_pattern = neurons['associative_cortex'].get_membrane_potentials()
            
            # Calculate similarity to target pattern
            similarity = self._calculate_similarity(current_pattern, self.target_pattern)
            self.progress = similarity
        
        self.last_progress_update = current_time
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between patterns."""
        if len(pattern1) != len(pattern2):
            # Handle size mismatch
            min_size = min(len(pattern1), len(pattern2))
            pattern1 = pattern1[:min_size]
            pattern2 = pattern2[:min_size]
        
        dot_product = np.dot(pattern1, pattern2)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)