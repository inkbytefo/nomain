## Developer: inkbytefo
## Modified: 2025-11-09

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import time
from collections import deque

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - using fallback language processing")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - template-based responses only")


class LanguageModule:
    """
    Module 4: Language Module - Text-Concept Interface
    
    Implements hybrid language processing combining:
    - Text-to-Concept: Sentence transformers for semantic understanding
    - Concept-to-Text: Template-based and optional LLM for response generation
    
    This bridges human language with the SNN's conceptual understanding.
    """
    
    def __init__(self, concept_neurons, associative_cortex, config: Dict[str, Any]):
        """
        Initialize Language Module.
        
        Args:
            concept_neurons: Language concept neuron population
            associative_cortex: Associative cortex neuron population
            config: Language module configuration
        """
        self.concept_neurons = concept_neurons
        self.associative_cortex = associative_cortex
        self.config = config
        
        # Model configuration
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vocab_size = config.get('vocab_size', 1000)
        self.concept_neurons_count = config.get('concept_neurons', 1000)
        
        # Response generation settings
        response_config = config.get('response_generation', {})
        self.template_based = response_config.get('template_based', True)
        self.llm_fallback = response_config.get('llm_fallback', False)
        self.llm_model_name = response_config.get('llm_model', 'gpt2')
        
        # Initialize models
        self.sentence_model = None
        self.llm_model = None
        self.tokenizer = None
        
        # SNN-LLM interface components
        self.autoencoder = None
        self.embedding_dim = 768  # Standard sentence transformer dimension
        self.concept_firing_rates = deque(maxlen=100)  # Store recent firing rates
        
        self._initialize_models()
        self._initialize_autoencoder()
        
        # Concept-word mappings
        self.concept_to_words = {}
        self.word_to_concepts = {}
        self.concept_embeddings = {}
        
        # Response templates
        self.response_templates = [
            "I see {concept}",
            "I understand {concept}",
            "That reminds me of {concept}",
            "I'm thinking about {concept}",
            "I notice {concept}",
            "I perceive {concept}"
        ]
        
        # Language state
        self.current_concepts = {}
        self.last_response_time = 0.0
        self.response_history = []
        
        logger.info("Language Module initialized")
    
    def _initialize_autoencoder(self):
        """Initialize autoencoder for SNN-LLM interface."""
        try:
            # Create a simple autoencoder for concept-embedding conversion
            # In a full implementation, this would be pre-trained on SNN activity patterns
            input_dim = self.concept_neurons_count
            
            # Simple linear autoencoder (could be replaced with neural network)
            self.encoder_weights = np.random.randn(self.embedding_dim, input_dim) * 0.1
            self.decoder_weights = np.random.randn(input_dim, self.embedding_dim) * 0.1
            
            # Normalize weights
            self.encoder_weights /= np.linalg.norm(self.encoder_weights, axis=1, keepdims=True)
            self.decoder_weights /= np.linalg.norm(self.decoder_weights, axis=0, keepdims=True)
            
            logger.info("Autoencoder initialized for SNN-LLM interface")
        except Exception as e:
            logger.error(f"Failed to initialize autoencoder: {e}")
            self.encoder_weights = None
            self.decoder_weights = None
    
    def _initialize_models(self):
        """Initialize language models based on availability."""
        # Initialize sentence transformer for text-to-concept
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.model_name)
                logger.info(f"Sentence transformer loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        
        # Initialize LLM for concept-to-text (optional)
        if self.llm_fallback and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_model_name)
                self.llm_model = transformers.AutoModelForCausalLM.from_pretrained(self.llm_model_name)
                logger.info(f"LLM loaded: {self.llm_model_name}")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                self.llm_model = None
                self.tokenizer = None
    
    def process_text_input(self, text: str) -> np.ndarray:
        """
        Process text input and convert to concept activation.
        
        Args:
            text: Input text string
            
        Returns:
            Concept activation array
        """
        if not text.strip():
            return np.zeros(self.concept_neurons_count)
        
        # Get semantic embedding
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text)
            except Exception as e:
                logger.error(f"Sentence encoding error: {e}")
                embedding = self._fallback_embedding(text)
        else:
            embedding = self._fallback_embedding(text)
        
        # Convert embedding to concept activation
        concept_activation = self._embedding_to_concepts(embedding)
        
        # Store current concepts
        self.current_concepts = {
            'text': text,
            'embedding': embedding,
            'concept_activation': concept_activation,
            'timestamp': time.time()
        }
        
        return concept_activation
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding method when sentence transformer unavailable.
        
        Args:
            text: Input text
            
        Returns:
            Simple embedding vector
        """
        # Simple word-based embedding
        words = text.lower().split()
        embedding = np.zeros(384)  # Default sentence transformer size
        
        for i, word in enumerate(words[:min(len(words), 50)]):
            # Simple hash-based embedding
            word_hash = hash(word) % 1000
            if i < len(embedding):
                embedding[i] = word_hash / 1000.0
        
        return embedding
    
    def _embedding_to_concepts(self, embedding: np.ndarray) -> np.ndarray:
        """
        Convert semantic embedding to concept neuron activation.
        
        Args:
            embedding: Semantic embedding vector
            
        Returns:
            Concept activation array
        """
        # Resize embedding to match concept neurons
        if len(embedding) != self.concept_neurons_count:
            # Simple interpolation/extrapolation
            if len(embedding) > self.concept_neurons_count:
                # Downsample
                indices = np.linspace(0, len(embedding) - 1, self.concept_neurons_count).astype(int)
                concept_activation = embedding[indices]
            else:
                # Upsample with repetition
                concept_activation = np.zeros(self.concept_neurons_count)
                for i in range(self.concept_neurons_count):
                    concept_activation[i] = embedding[i % len(embedding)]
        else:
            concept_activation = embedding.copy()
        
        # Normalize and scale to firing rates
        concept_activation = np.abs(concept_activation)
        if np.max(concept_activation) > 0:
            concept_activation = concept_activation / np.max(concept_activation)
        
        # Convert to spike rates (0-200 Hz)
        concept_activation = concept_activation * 200.0
        
        return concept_activation
    
    def generate_response(self, neurons: Dict[str, Any], simulation_time: float) -> Optional[str]:
        """
        Generate language response from current neural state using SNN-LLM interface.
        
        Args:
            neurons: Dictionary of neuron populations
            simulation_time: Current simulation time
            
        Returns:
            Generated response string or None
        """
        # Rate limiting for responses
        if simulation_time - self.last_response_time < 2.0:
            return None
        
        # Get current concept activations from associative cortex (primary source)
        if 'associative_cortex' in neurons:
            assoc_activations = neurons['associative_cortex'].get_membrane_potentials()
            assoc_spikes = neurons['associative_cortex'].get_spikes()
            
            # Calculate firing rates from recent activity
            firing_rates = self._calculate_firing_rates(assoc_activations, assoc_spikes)
            self.concept_firing_rates.append(firing_rates)
        else:
            return None
        
        # Convert SNN pattern to embedding using autoencoder
        concept_embedding = self.concepts_to_embedding(firing_rates)
        
        # Get language concept activations if available
        if 'language_concepts' in neurons:
            lang_spikes = neurons['language_concepts'].get_spikes()
            lang_activations = neurons['language_concepts'].get_membrane_potentials()
            active_concepts = self._get_active_concepts(lang_activations, lang_spikes)
        else:
            active_concepts = []
        
        if not active_concepts and np.sum(firing_rates) < 0.1:
            return None
        
        # Generate response based on concept embedding and active concepts
        response = self._generate_embedding_response(concept_embedding, active_concepts)
        
        if response:
            self.last_response_time = simulation_time
            self.response_history.append({
                'response': response,
                'embedding_norm': np.linalg.norm(concept_embedding),
                'concepts': active_concepts,
                'timestamp': simulation_time
            })
        
        return response
    
    def _calculate_firing_rates(self, activations: np.ndarray, spikes: np.ndarray,
                             window_size: float = 0.1) -> np.ndarray:
        """
        Calculate approximate firing rates from activations and spikes.
        
        Args:
            activations: Membrane potentials
            spikes: Current spike activity
            window_size: Time window for rate calculation
            
        Returns:
            Firing rate estimates
        """
        # Simple approximation: combine activation level with spike activity
        # In a full implementation, this would use actual spike timing history
        firing_rates = np.abs(activations) + spikes * 10.0  # Spikes contribute more to rate
        
        # Normalize to reasonable firing rate range (0-100 Hz)
        if np.max(firing_rates) > 0:
            firing_rates = firing_rates / np.max(firing_rates) * 100.0
        
        return firing_rates
    
    def _get_active_concepts(self, activations: np.ndarray, spikes: np.ndarray, 
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get currently active concepts.
        
        Args:
            activations: Neuron activation potentials
            spikes: Spike activity
            threshold: Activation threshold
            
        Returns:
            List of active concept information
        """
        active_concepts = []
        
        # Find neurons above threshold
        active_indices = np.where(activations > threshold)[0]
        
        for idx in active_indices:
            concept_info = {
                'neuron_id': int(idx),
                'activation': float(activations[idx]),
                'spiking': bool(spikes[idx] > 0),
                'concept_word': self._neuron_to_word(idx)
            }
            active_concepts.append(concept_info)
        
        # Sort by activation strength
        active_concepts.sort(key=lambda x: x['activation'], reverse=True)
        
        return active_concepts[:5]  # Return top 5 concepts
    
    def _neuron_to_word(self, neuron_id: int) -> str:
        """
        Convert neuron ID to representative word.
        
        Args:
            neuron_id: Neuron index
            
        Returns:
            Representative word
        """
        if neuron_id in self.concept_to_words:
            return self.concept_to_words[neuron_id]
        
        # Generate word based on neuron ID (simplified)
        word_map = {
            0: "something", 1: "object", 2: "pattern", 3: "shape", 4: "color",
            5: "movement", 6: "sound", 7: "light", 8: "dark", 9: "change",
            10: "familiar", 11: "new", 12: "interesting", 13: "simple", 14: "complex"
        }
        
        word = word_map.get(neuron_id % len(word_map), f"concept_{neuron_id}")
        self.concept_to_words[neuron_id] = word
        return word
    
    def _generate_concept_response(self, active_concepts: List[Dict[str, Any]], 
                                 context: np.ndarray) -> str:
        """
        Generate response based on active concepts.
        
        Args:
            active_concepts: List of active concept information
            context: Associative cortex context
            
        Returns:
            Generated response
        """
        if not active_concepts:
            return None
        
        # Get primary concept
        primary_concept = active_concepts[0]
        concept_word = primary_concept['concept_word']
        
        # Template-based response
        if self.template_based:
            template = np.random.choice(self.response_templates)
            response = template.format(concept=concept_word)
            
            # Add context if available
            if len(active_concepts) > 1:
                secondary_concepts = [c['concept_word'] for c in active_concepts[1:3]]
                if secondary_concepts:
                    response += f" and {', '.join(secondary_concepts)}"
            
            return response
        
        # LLM-based response (if available)
        elif self.llm_model and self.tokenizer:
            return self._generate_llm_response(active_concepts, context)
        
        return None
    
    def _generate_embedding_response(self, embedding: np.ndarray,
                                   active_concepts: List[Dict[str, Any]]) -> str:
        """
        Generate response based on concept embedding.
        
        Args:
            embedding: Concept embedding vector
            active_concepts: Active concepts for context
            
        Returns:
            Generated response
        """
        # Template-based response using embedding similarity
        if self.template_based and active_concepts:
            # Get primary concept
            primary_concept = active_concepts[0]['concept_word']
            
            # Use embedding to select appropriate response template
            embedding_norm = np.linalg.norm(embedding)
            
            if embedding_norm > 0.7:
                # High activation - more confident response
                templates = [
                    f"I clearly perceive {primary_concept}",
                    f"I'm strongly aware of {primary_concept}",
                    f"{primary_concept.capitalize()} is very present in my experience"
                ]
            elif embedding_norm > 0.4:
                # Medium activation - moderate response
                templates = [
                    f"I notice {primary_concept}",
                    f"I'm thinking about {primary_concept}",
                    f"{primary_concept} seems relevant"
                ]
            else:
                # Low activation - tentative response
                templates = [
                    f"I might be sensing {primary_concept}",
                    f"There's something like {primary_concept}",
                    f"I'm not sure, but perhaps {primary_concept}"
                ]
            
            response = np.random.choice(templates)
            
            # Add secondary concepts if available
            if len(active_concepts) > 1:
                secondary_concepts = [c['concept_word'] for c in active_concepts[1:3]]
                if secondary_concepts:
                    response += f", along with {', '.join(secondary_concepts)}"
            
            return response + "."
        
        # LLM-based response (if available)
        elif self.llm_model and self.tokenizer:
            return self._generate_llm_response(active_concepts, embedding)
        
        return None
    
    def _generate_llm_response(self, active_concepts: List[Dict[str, Any]],
                             embedding: np.ndarray) -> str:
        """
        Generate response using LLM with embedding context.
        
        Args:
            active_concepts: Active concepts
            embedding: Concept embedding for context
            
        Returns:
            LLM-generated response
        """
        try:
            # Create prompt from concepts and embedding state
            concept_words = [c['concept_word'] for c in active_concepts[:3]]
            embedding_state = "highly activated" if np.linalg.norm(embedding) > 0.6 else "moderately activated"
            
            prompt = f"I am experiencing {embedding_state} concepts: {', '.join(concept_words)}. This makes me feel"
            
            # Generate response
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.llm_model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Fallback to template
            if active_concepts:
                return self.response_templates[0].format(concept=active_concepts[0]['concept_word'])
            return "I'm processing some internal concepts."
    
    def concepts_to_text(self, concept_pattern: np.ndarray) -> str:
        """
        Convert concept pattern back to text description.
        
        Args:
            concept_pattern: Concept activation pattern
            
        Returns:
            Text description
        """
        # Find most active concepts
        active_indices = np.argsort(concept_pattern)[-5:][::-1]
        
        words = []
        for idx in active_indices:
            if concept_pattern[idx] > 0.1:
                words.append(self._neuron_to_word(idx))
        
        if words:
            return f"I'm thinking about: {', '.join(words)}"
        else:
            return "I'm not processing any specific concepts right now"
    
    def get_language_state(self) -> Dict[str, Any]:
        """Get current language module state."""
        return {
            'current_concepts': self.current_concepts,
            'response_count': len(self.response_history),
            'last_response_time': self.last_response_time,
            'models_loaded': {
                'sentence_transformer': self.sentence_model is not None,
                'llm': self.llm_model is not None
            }
        }
    
    def reset(self):
        """Reset language module state."""
        self.current_concepts = {}
        self.last_response_time = 0.0
        self.response_history.clear()
        logger.info("Language Module reset")