## Developer: inkbytefo
## Modified: 2025-11-08

# PSINet Oracle Loop - Milestone 11: The Grand Unification

## Overview

The PSINet Oracle Loop represents the culmination of Project Chimera - a multi-threaded AGI system that bridges simulation to reality through real-time sensory processing, neural simulation, and grounded language communication.

## Architecture

The Oracle Loop consists of three synchronized threads:

### Thread 1: PSINet Core
- **Purpose**: Continuous neural simulation using Brian2
- **Function**: Runs the hierarchical spiking neural network in 10ms chunks
- **Integration**: Processes sensory input and generates language output

### Thread 2: Sensory Processing
- **Purpose**: Real-time sensory data ingestion
- **Components**: 
  - VideoToSpikes: DVS emulation using OpenCV
  - AudioToSpikes: Cochleagram generation using FFT
- **Output**: Poisson spike trains for neural input

### Thread 3: Dialogue Interface
- **Purpose**: User interaction and language processing
- **Features**: 
  - Text-to-concepts encoding
  - Concepts-to-text generation
  - Real-time chat interface

## Key Components

### SharedState
Inter-thread communication container with:
- Thread-safe queues for data exchange
- Synchronization locks
- Performance statistics tracking

### Language System
- **TextToConcepts**: Maps user input to neural activations
- **ConceptsToText**: Generates responses from neural activity
- **Curiosity-driven**: Responds based on prediction errors

### Real-time Encoders
- **VideoToSpikes**: Dynamic Vision Sensor emulation
- **AudioToSpikes**: Frequency-domain audio processing
- **Biologically plausible**: Poisson firing rate encoding

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_oracle_loop.py
```

## Usage

### Basic Operation
```bash
# Run with full sensory input
python run_agi.py

# Run without camera
python run_agi.py --no-camera

# Run without audio
python run_agi.py --no-audio

# Run with custom configuration
python run_agi.py --config configs/language_enabled_realtime.yaml
```

### Interactive Commands
- `hello` - Greet PSINet
- `stats` - Show system statistics
- `help` - Display help message
- `quit` - Shutdown the system

## Configuration

The system uses YAML configuration files:

```yaml
simulation_params:
  chunk_duration_ms: 10
  max_duration_seconds: 60

input_params:
  source: realtime
  realtime_config:
    enable_video: true
    enable_audio: true
    video_resolution: [32, 32]
    audio_sample_rate: 16000

language_params:
  enable_language: true
  vocab_size: 100
  concept_neurons: 50
  curiosity_threshold: 0.7
```

## Testing

```bash
# Run all tests
python test_oracle_loop.py

# Run specific test classes
python -m unittest test_oracle_loop.TestSharedState
python -m unittest test_oracle_loop.TestPSINetCoreThread
python -m unittest test_oracle_loop.TestSensoryThread
python -m unittest test_oracle_loop.TestDialogueThread
```

## Performance

### System Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: USB webcam (optional)
- **Microphone**: Audio input device (optional)

### Benchmarks
- **Sensory Processing**: ~30 FPS video, 44.1 kHz audio
- **Neural Simulation**: 10ms chunks with 1000+ neurons
- **Language Response**: <1 second latency
- **Memory Usage**: <500MB typical

## File Structure

```
PSINet-nomain/
├── run_agi.py                    # Main Oracle Loop application
├── test_oracle_loop.py           # Comprehensive test suite
├── configs/
│   └── language_enabled_realtime.yaml  # AGI configuration
├── psinet/
│   ├── io/
│   │   └── realtime_encoders.py  # Sensory processing
│   ├── language/
│   │   ├── input.py              # Text-to-concepts
│   │   └── output.py             # Concepts-to-text
│   └── simulation/
│       └── simulator.py          # Neural simulation
└── README_ORACLE_LOOP.md         # This documentation
```

## Development

### Adding New Features
1. Extend SharedState for new data channels
2. Implement thread-safe communication
3. Add corresponding tests
4. Update configuration schema

### Debugging
```bash
# Enable debug logging
python run_agi.py --log-level DEBUG

# Run with minimal configuration
python run_agi.py --no-camera --no-audio
```

### Performance Tuning
- Adjust `chunk_duration_ms` for latency vs. accuracy
- Modify network size in configuration
- Tune sensory processing parameters

## Research Applications

The Oracle Loop enables research in:
- **Embodied cognition**: Real-world sensory grounding
- **Language acquisition**: Word-concept association learning
- **Predictive coding**: Error-driven neural processing
- **Multi-modal integration**: Audio-visual synthesis
- **Human-AI interaction**: Natural language dialogue

## Limitations

- **Sensory fidelity**: Low-resolution video/audio
- **Language complexity**: Simple vocabulary and grammar
- **Neural scale**: Limited to thousands of neurons
- **Real-time constraints**: Processing latency affects performance

## Future Work

- **Enhanced sensory**: Higher resolution, more modalities
- **Larger networks**: Hierarchical expansion
- **Advanced language**: Grammar, syntax, semantics
- **Learning algorithms**: Online adaptation, memory
- **Hardware acceleration**: GPU/TPU optimization

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This project is part of PSINet research initiative. See LICENSE file for details.

## Acknowledgments

- Brian2 neural simulation framework
- OpenCV computer vision library
- NumPy scientific computing
- YAML configuration management

---

**PSINet Oracle Loop - Bridging Simulation to Reality Through Multi-Threaded AGI**