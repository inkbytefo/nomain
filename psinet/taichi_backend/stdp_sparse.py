## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
from typing import Optional
from scipy.sparse import csr_matrix

@ti.data_oriented
class STDPSparseKernel:
    def __init__(self, pre_neurons: int, post_neurons: int, dt: float = 0.001, sparsity: float = 0.9):
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.dt = dt
        self.sparsity = sparsity
        
        self.weights = ti.field(dtype=ti.f32, shape=(pre_neurons, post_neurons))
        self.apre = ti.field(dtype=ti.f32, shape=pre_neurons)
        self.apost = ti.field(dtype=ti.f32, shape=post_neurons)
        self.pre_spikes = ti.field(dtype=ti.i32, shape=pre_neurons)
        self.post_spikes = ti.field(dtype=ti.i32, shape=post_neurons)
        
        self.tau_pre = ti.field(dtype=ti.f32, shape=())
        self.tau_post = ti.field(dtype=ti.f32, shape=())
        self.a_plus = ti.field(dtype=ti.f32, shape=())
        self.a_minus = ti.field(dtype=ti.f32, shape=())
        self.w_max = ti.field(dtype=ti.f32, shape=())
        self.dopamine = ti.field(dtype=ti.f32, shape=())
        
        # Output buffer
        self.psc_output = ti.field(dtype=ti.f32, shape=post_neurons)
        
        self._init_parameters()
        self._init_weights()
    
    def _init_parameters(self):
        self.tau_pre[None] = 0.02
        self.tau_post[None] = 0.02
        self.a_plus[None] = 0.01
        self.a_minus[None] = -0.0105
        self.w_max[None] = 0.01
        self.dopamine[None] = 1.0
        self._reset_traces()
    
    @ti.kernel
    def _init_weights(self):
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if ti.random() < self.sparsity:
                self.weights[i, j] = 0.0
            else:
                self.weights[i, j] = ti.random() * 0.01
    
    @ti.kernel
    def _reset_traces(self):
        for i in range(self.pre_neurons):
            self.apre[i] = 0.0
            self.pre_spikes[i] = 0
        for j in range(self.post_neurons):
            self.apost[j] = 0.0
            self.post_spikes[j] = 0
            self.psc_output[j] = 0.0
    
    @ti.kernel
    def step(self, pre_spikes_in: ti.types.ndarray(ndim=1, dtype=ti.i32),
                   post_spikes_in: ti.types.ndarray(ndim=1, dtype=ti.i32)):
        # Copy input spikes
        for i in range(self.pre_neurons):
            self.pre_spikes[i] = pre_spikes_in[i]
        for j in range(self.post_neurons):
            self.post_spikes[j] = post_spikes_in[j]
        
        # Update traces
        for i in range(self.pre_neurons):
            self.apre[i] *= ti.exp(-self.dt / self.tau_pre[None])
            if self.pre_spikes[i]: self.apre[i] += 1.0
        for j in range(self.post_neurons):
            self.apost[j] *= ti.exp(-self.dt / self.tau_post[None])
            if self.post_spikes[j]: self.apost[j] += 1.0
        
        # Update weights + dopamine modulation
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0:
                if self.pre_spikes[i] and self.apost[j] > 0:
                    self.weights[i, j] += self.a_plus[None] * self.apost[j]
                if self.post_spikes[j] and self.apre[i] > 0:
                    eff_ltd = self.a_minus[None] * (2.0 - self.dopamine[None])
                    self.weights[i, j] += eff_ltd * self.apre[i]
                self.weights[i, j] = ti.max(0.0, ti.min(self.w_max[None], self.weights[i, j]))
        
        # Compute PSC
        for j in range(self.post_neurons):
            self.psc_output[j] = 0.0
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0 and pre_spikes_in[i]:
                self.psc_output[j] += self.weights[i, j]
    
    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> np.ndarray:
        pre_in = pre_spikes.astype(np.int32)
        post_in = post_spikes.astype(np.int32)
        self.step(pre_in, post_in)
        return self.psc_output.to_numpy()
    
    def set_dopamine(self, level: float):
        self.dopamine[None] = np.clip(level, 0.0, 2.0)
    
    def get_weights(self) -> np.ndarray:
        return self.weights.to_numpy()
    
    def get_weight_stats(self) -> dict:
        w = self.weights.to_numpy()
        nz = w[w > 0]
        return {
            'mean_weight': float(nz.mean()) if len(nz) > 0 else 0.0,
            'sparsity': 1.0 - len(nz) / w.size,
            'total_connections': len(nz)
        }
    
    def get_trace_stats(self) -> dict:
        apre_np = self.apre.to_numpy()
        apost_np = self.apost.to_numpy()
        
        return {
            'pre_trace_mean': float(np.mean(apre_np)),
            'pre_trace_max': float(np.max(apre_np)),
            'post_trace_mean': float(np.mean(apost_np)),
            'post_trace_max': float(np.max(apost_np)),
            'dopamine_level': float(self.dopamine[None])
        }
    
    def reset(self):
        """Reset STDP traces and spikes."""
        self._reset_traces()
    
    def set_parameters(self, tau_pre: Optional[float] = None,
                      tau_post: Optional[float] = None,
                      a_plus: Optional[float] = None,
                      a_minus: Optional[float] = None,
                      w_max: Optional[float] = None):
        """Update STDP parameters."""
        if tau_pre is not None:
            self.tau_pre[None] = tau_pre
        if tau_post is not None:
            self.tau_post[None] = tau_post
        if a_plus is not None:
            self.a_plus[None] = a_plus
        if a_minus is not None:
            self.a_minus[None] = a_minus
        if w_max is not None:
            self.w_max[None] = w_max
    
    def get_sparse_weights(self) -> csr_matrix:
        """Get weight matrix as scipy sparse CSR matrix."""
        dense_weights = self.get_weights()
        return csr_matrix(dense_weights)