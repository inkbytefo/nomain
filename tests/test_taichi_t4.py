## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=15, debug=False)

from psinet.core.taichi_neuron import BionicNeuron
from psinet.core.taichi_synapse import BionicSynapse
import time
import numpy as np

print(f"Taichi CUDA backend: {ti.cfg.arch}")

# 10k nöron test
pre = BionicNeuron(10000, dt=0.001, sparsity=0.9)
post = BionicNeuron(8000, dt=0.001, sparsity=0.9)
syn = BionicSynapse(pre, post, sparsity=0.9)

# Rastgele input
input_current = np.random.uniform(0, 2, 10000).astype(np.float32)

# Dopamin test
syn.set_dopamine(0.5)  # LTD artmalı
syn.set_dopamine(1.5)  # LTD azalmalı

# 100 timestep
start = time.time()
for _ in range(100):
    pre.update(input_current)
    pre_spikes = pre.get_spikes()
    post_spikes = post.get_spikes()
    psc = syn.update(pre_spikes, post_spikes)
    post.update(psc)
end = time.time()

duration = end - start
fps = 100 / duration

print(f"100 timestep sürdü: {duration:.4f} sn")
print(f"FPS: {fps:.1f}")
print(f"Spike rate: {pre.get_spike_count() / 10000:.1f} Hz")
print(f"Mean weight: {syn.get_weight_stats()['mean_weight']:.6f}")
print(f"Taichi T4 ÇALIŞIYOR! Proto-AGI doğdu!")