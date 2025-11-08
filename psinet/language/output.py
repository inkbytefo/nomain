## Developer: inkbytefo
## Modified: 2025-11-08

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
import time

logger = logging.getLogger(__name__)


class ConceptsToText:
    """
    Converts neural activity patterns to text output using template-based generation.
    This is the only way PSINet can act on the world through language.
    """
    
    def __init__(self, text_to_concepts, curiosity_threshold: float = 0.7,
                 curiosity_duration: float = 2.0):
        """
        Initialize the concepts-to-text converter.
        
        Args:
            text_to_concepts: Reference to TextToConcepts instance for vocabulary access
            curiosity_threshold: Activation threshold for triggering curiosity
            curiosity_duration: Time in seconds that high error must persist
        """
        self.text_to_concepts = text_to_concepts
        self.curiosity_threshold = curiosity_threshold
        self.curiosity_duration = curiosity_duration
        
        # Sentence templates
        self.templates = {
            'observation': [
                "I see a {object}.",
                "I can see {object}.",
                "There is a {object}.",
                "I notice {object}."
            ],
            'property': [
                "The {object} is {property}.",
                "I see the {object} is {property}.",
                "The {object} looks {property}.",
                "That {object} appears {property}."
            ],
            'curiosity': [
                "What is that?",
                "What am I seeing?",
                "What is this?",
                "Can you tell me what this is?"
            ],
            'action': [
                "I want to {action}.",
                "I should {action}.",
                "Let me {action}.",
                "I need to {action}."
            ]
        }
        
        # Concept categories for template selection
        self.object_words = {
            'cup', 'ball', 'book', 'phone', 'key', 'pen', 'paper', 'computer',
            'table', 'chair', 'door', 'window', 'light', 'bottle', 'glass',
            'fin', 'top', 'kitap', 'telefon', 'anahtar', 'kalem', 'kağıt', 'bilgisayar',
            'masa', 'sandalye', 'kapı', 'pencere', 'ışık', 'şişe', 'bardak'
        }
        
        self.property_words = {
            'big', 'small', 'red', 'blue', 'green', 'white', 'black', 'yellow',
            'hot', 'cold', 'heavy', 'light', 'soft', 'hard', 'smooth', 'rough',
            'büyük', 'küçük', 'kırmızı', 'mavi', 'yeşil', 'beyaz', 'siyah', 'sarı',
            'sıcak', 'soğuk', 'ağır', 'hafif', 'yumuşak', 'sert', 'pürüzsüz', 'kaba'
        }
        
        self.action_words = {
            'see', 'hear', 'touch', 'move', 'stop', 'go', 'take', 'put',
            'gör', 'duy', 'dokun', 'hareket', 'dur', 'git', 'al', 'koy'
        }
        
        # State tracking
        self.recent_concepts = deque(maxlen=10)  # Store recent concept activations
        self.last_output_time = 0.0
        self.output_cooldown = 0.5  # Minimum seconds between outputs
        
        # Curiosity tracking
        self.high_error_start_time = None
        self.curiosity_triggered = False
        
        # Output history
        self.output_history = []
        
    def generate_response(self, concept_activations: Dict[str, float], 
                         error_signal: float = 0.0, current_time: float = None) -> Optional[str]:
        """
        Generate text response based on current concept activations and error signals.
        
        Args:
            concept_activations: Dictionary of concept names to activation levels
            error_signal: Current error/prediction error signal (0-1)
            current_time: Current simulation time
            
        Returns:
            Generated text response or None if no appropriate response
        """
        if current_time is None:
            current_time = time.time()
        
        # Check output cooldown
        if current_time - self.last_output_time < self.output_cooldown:
            return None
        
        # Store recent concepts
        self.recent_concepts.append(concept_activations.copy())
        
        # Check for curiosity trigger first (highest priority)
        curiosity_response = self._check_curiosity_trigger(error_signal, current_time)
        if curiosity_response:
            return curiosity_response
        
        # Check for stable representations
        stable_response = self._check_stable_representations(concept_activations, current_time)
        if stable_response:
            return stable_response
        
        return None
    
    def _check_curiosity_trigger(self, error_signal: float, current_time: float) -> Optional[str]:
        """
        Check if curiosity should be triggered based on sustained high error.
        
        Args:
            error_signal: Current error signal (0-1)
            current_time: Current time
            
        Returns:
            Curiosity response or None
        """
        if error_signal > self.curiosity_threshold:
            if self.high_error_start_time is None:
                self.high_error_start_time = current_time
                logger.debug(f"High error detected: {error_signal:.3f}")
            elif (current_time - self.high_error_start_time) >= self.curiosity_duration:
                if not self.curiosity_triggered:
                    self.curiosity_triggered = True
                    response = self._generate_curiosity_response()
                    self._record_output(response, current_time, 'curiosity')
                    return response
        else:
            # Reset curiosity tracking
            self.high_error_start_time = None
            self.curiosity_triggered = False
        
        return None
    
    def _check_stable_representations(self, concept_activations: Dict[str, float], 
                                   current_time: float) -> Optional[str]:
        """
        Check for stable concept representations that should trigger verbal responses.
        
        Args:
            concept_activations: Current concept activations
            current_time: Current time
            
        Returns:
            Generated response or None
        """
        # Filter for significant activations - lowered threshold
        significant_concepts = {k: v for k, v in concept_activations.items() if v > 0.1}
        
        if not significant_concepts:
            return None
        
        # DEĞİŞİKLİK: Doğrudan anlamlı konseptlerle yanıt üret, stabilite bekleme
        return self._generate_template_response(significant_concepts, current_time)
    
    def _find_stable_concepts(self, current_concepts: Dict[str, float]) -> Dict[str, float]:
        """
        Find concepts that have been consistently active across recent time windows.
        
        Args:
            current_concepts: Current significant concept activations
            
        Returns:
            Dictionary of stable concepts with average activation
        """
        if len(self.recent_concepts) < 3:
            return {}
        
        stable_concepts = {}
        
        # Check each concept in current activations
        for concept, current_activation in current_concepts.items():
            activations = []
            
            # Collect activation history for this concept
            for past_concepts in list(self.recent_concepts)[-3:]:
                past_activation = past_concepts.get(concept, 0.0)
                activations.append(past_activation)
            
            activations.append(current_activation)
            
            # Check if consistently active (low variance)
            if len(activations) >= 3:
                mean_activation = np.mean(activations)
                activation_variance = np.var(activations)
                
                # Consider stable if mean > 0.2 and variance < 0.2
                if mean_activation > 0.2 and activation_variance < 0.2:
                    stable_concepts[concept] = mean_activation
        
        return stable_concepts
    
    def _generate_template_response(self, stable_concepts: Dict[str, float], 
                                  current_time: float) -> Optional[str]:
        """
        Generate a response using appropriate templates based on stable concepts.
        
        Args:
            stable_concepts: Dictionary of stable concept activations
            current_time: Current time
            
        Returns:
            Generated response or None
        """
        # Categorize concepts
        objects = [c for c in stable_concepts.keys() if c in self.object_words]
        properties = [c for c in stable_concepts.keys() if c in self.property_words]
        actions = [c for c in stable_concepts.keys() if c in self.action_words]
        
        # Prioritize responses: object+property > object > action
        if objects and properties:
            # Object + property response
            object_word = max(objects, key=lambda x: stable_concepts[x])
            property_word = max(properties, key=lambda x: stable_concepts[x])
            
            template = np.random.choice(self.templates['property'])
            response = template.format(object=object_word, property=property_word)
            
        elif objects:
            # Object-only response
            object_word = max(objects, key=lambda x: stable_concepts[x])
            
            template = np.random.choice(self.templates['observation'])
            response = template.format(object=object_word)
            
        elif actions:
            # Action response
            action_word = max(actions, key=lambda x: stable_concepts[x])
            
            template = np.random.choice(self.templates['action'])
            response = template.format(action=action_word)
            
        else:
            return None
        
        self._record_output(response, current_time, 'template')
        return response
    
    def _generate_curiosity_response(self) -> str:
        """Generate a curiosity-based question."""
        template = np.random.choice(self.templates['curiosity'])
        return template
    
    def _record_output(self, response: str, current_time: float, response_type: str):
        """Record the output for tracking and analysis."""
        self.output_history.append({
            'text': response,
            'time': current_time,
            'type': response_type
        })
        self.last_output_time = current_time
        
        logger.info(f"Generated {response_type} response: '{response}'")
    
    def add_template(self, category: str, template: str):
        """
        Add a new template to the specified category.
        
        Args:
            category: Template category ('observation', 'property', 'curiosity', 'action')
            template: Template string with placeholders
        """
        if category not in self.templates:
            self.templates[category] = []
        
        self.templates[category].append(template)
        logger.info(f"Added template to {category}: '{template}'")
    
    def add_word_category(self, word: str, category: str):
        """
        Add a word to a specific category.
        
        Args:
            word: Word to add
            category: Category name ('object', 'property', 'action')
        """
        if category == 'object':
            self.object_words.add(word.lower())
        elif category == 'property':
            self.property_words.add(word.lower())
        elif category == 'action':
            self.action_words.add(word.lower())
        else:
            logger.warning(f"Unknown category: {category}")
    
    def get_output_stats(self) -> Dict[str, int]:
        """Get statistics about generated outputs."""
        stats = {'total': len(self.output_history)}
        
        for output in self.output_history:
            response_type = output['type']
            stats[response_type] = stats.get(response_type, 0) + 1
        
        return stats
    
    def get_recent_outputs(self, count: int = 5) -> List[Dict]:
        """Get the most recent outputs."""
        return self.output_history[-count:] if self.output_history else []
    
    def reset_history(self):
        """Clear output history and reset state."""
        self.output_history.clear()
        self.recent_concepts.clear()
        self.last_output_time = 0.0
        self.high_error_start_time = None
        self.curiosity_triggered = False
        
        logger.info("Reset language output history and state")
    
    def set_curiosity_parameters(self, threshold: float = None, duration: float = None):
        """
        Update curiosity trigger parameters.
        
        Args:
            threshold: New curiosity threshold (0-1)
            duration: New curiosity duration in seconds
        """
        if threshold is not None:
            self.curiosity_threshold = threshold
        if duration is not None:
            self.curiosity_duration = duration
        
        logger.info(f"Updated curiosity parameters: threshold={self.curiosity_threshold}, "
                   f"duration={self.curiosity_duration}s")
    
    def force_response(self, response_type: str, **kwargs) -> str:
        """
        Force generate a response of a specific type (for testing/debugging).
        
        Args:
            response_type: Type of response to generate
            **kwargs: Additional parameters for template filling
            
        Returns:
            Generated response
        """
        if response_type not in self.templates:
            raise ValueError(f"Unknown response type: {response_type}")
        
        template = np.random.choice(self.templates[response_type])
        
        try:
            response = template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template parameter: {e}")
            response = template  # Return unfilled template
        
        return response