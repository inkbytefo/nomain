# Configuration Files

This directory contains YAML configurations for running PSINet simulations via the centralized Simulator.

- mnist_multi_pattern_0_vs_1.yaml
  - Purpose: prototyping and quick iterations on the 0 vs 1 discrimination task using the Python runtime backend.
- mnist_multi_pattern_performance.yaml
  - Purpose: longer, performance-tuned runs using cpp_standalone and more cycles.

Key parameters (schema overview):
- run_id: Short name for the run; used in output directory naming.
- simulation_params:
  - brian2_device: "runtime" for pure Python or "cpp_standalone" for compiled runs.
  - duration_per_pattern_ms: Presentation time per pattern.
  - silence_period_ms: Silence window between patterns.
  - cycles: How many (0→silence→1→silence) cycles to simulate.
- input_params:
  - dataset: "mnist" to use the torchvision MNIST loader (with caching) or "synthetic" to use built-in digits.
  - data_dir: Cache directory for datasets (default: ~/.psinet_data).
  - patterns_to_learn: Digits to present (e.g., [0, 1]).
  - image_indices: Per-digit selection index within the dataset to vary exemplars.
  - max_rate_hz: Max firing rate used when converting images to Poisson rates.
- network_params:
  - num_excitatory_l1: Number of excitatory neurons in L1.
  - num_inhibitory_l1: Number of inhibitory neurons in L1.
  - enable_lateral_inhibition: Enables competitive interactions within L1.
  - lateral_strength: Strength of lateral inhibitory effect between excitatory neurons.
- learning_params:
  - w_max_inp_l1: Max synaptic weight for input→L1 synapses.
  - a_plus_inp_l1: LTP factor for STDP.
  - a_minus_inp_l1: LTD factor for STDP.
- monitor_params:
  - record_l1_spikes: Record spikes from L1 excitatory neurons.
  - record_input_spikes: Record spikes from input neurons.
  - record_weight_subset_size: Number of input→L1 synapses to sample for weight tracking.
- output_params:
  - base_output_dir: Base directory for outputs (default: outputs/).
  - save_plots: Save summary plots.
  - save_raw_data: Save raw numpy arrays.
