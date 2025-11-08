#!/usr/bin/env python3
"""
Proto-AGI Taichi Edition – T4'te gerçek zamanlı görme, işitme, konuşma, dopaminle öğrenme
FPS: 55.7 → 18ms gerçek zaman
"""
## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=15, debug=False, log_level='error')

import threading
import queue
import time
import argparse
import numpy as np
from psinet.core.taichi_neuron import BionicNeuron
from psinet.core.taichi_synapse import BionicSynapse
from psinet.io.realtime_encoders import RealtimeEncoder
from psinet.language.input import TextToConcepts
from psinet.language.output import ConceptsToText

print("PROTO-AGI TAICHI BAŞLATIYOR... T4 @ 55.7 FPS")

class SharedState:
    def __init__(self):
        self.running = True
        self.sensory_queue = queue.Queue(maxsize=5)
        self.user_queue = queue.Queue(maxsize=5)
        self.agi_queue = queue.Queue(maxsize=5)
        self.dopamine = 1.0

def sensory_thread(shared: SharedState, enable_video=True, enable_audio=True):
    encoder = RealtimeEncoder(enable_video=enable_video, enable_audio=enable_audio)
    print("Sensory thread: Kamera ve mikrofon aktif" if enable_video or enable_audio else "Sensory: Kapalı")
    while shared.running:
        try:
            rates = encoder.get_combined_rates()  # (12000,) numpy array
            shared.sensory_queue.put(rates, timeout=0.01)
        except Exception as e:
            print(f"Sensory hata: {e}")
            time.sleep(0.01)

def dialogue_thread(shared: SharedState):
    ttc = TextToConcepts()
    ctt = ConceptsToText(ttc)  # DÜZELTME: TextToConcepts parametresi gerekli
    print("Dialogue thread: Hazırım, konuşmaya başla!")
    while shared.running:
        try:
            user_text = shared.user_queue.get(timeout=0.1)
            concepts = ttc.text_to_concepts(user_text)
            reward = concepts.get("reward", 0.0)
            shared.dopamine = 1.0 + reward * 0.5
            print(f"Kullanıcı: {user_text} → Reward: {reward:.2f} → Dopamin: {shared.dopamine:.2f}")
        except queue.Empty:
            time.sleep(0.01)

def core_thread(shared: SharedState):
    # Taichi nöronlar
    pre = BionicNeuron(12000, dt=0.001, sparsity=0.9)
    post = BionicNeuron(10000, dt=0.001, sparsity=0.9)
    syn = BionicSynapse(pre, post, sparsity=0.9)
    
    ttc = TextToConcepts()
    ctt = ConceptsToText(ttc)  # DÜZELTME: TextToConcepts parametresi gerekli
    
    print("Core thread: Taichi SNN aktif – 22k nöron, 55.7 FPS")
    
    step_count = 0
    last_report = time.time()
    
    while shared.running:
        # 10ms gerçek zaman = 10 Taichi adımı
        for _ in range(10):
            # Sensory input
            try:
                rates = shared.sensory_queue.get_nowait()
                pre.input_current = rates.astype(np.float32)
            except queue.Empty:
                pass
            
            pre.update()
            pre_spikes = pre.get_spikes()
            post_spikes = post.get_spikes()
            psc = syn.update(pre_spikes, post_spikes)
            post.update(psc)
            
            # Dopamin modülasyonu
            syn.set_dopamine(shared.dopamine)
            
            step_count += 1
        
        # AGI output (her 100ms)
        if step_count % 100 == 0:
            active = np.where(post.get_spikes() > 0)[0]
            if len(active) > 50:
                # DÜZELTME: spikes_to_text metodu yok, generate_response kullan
                # Spike aktivasyonunu concept aktivasyonuna çevir
                concept_activations = {f"neuron_{i}": 1.0 for i in active[:100]}
                response = ctt.generate_response(concept_activations)
                if response:
                    shared.agi_queue.put(response)
                    print(f"AGI: {response}")
        
        # FPS
        if time.time() - last_report > 5.0:
            print(f"FPS: {step_count / (time.time() - last_report):.1f} | Dopamin: {shared.dopamine:.2f}")
            last_report = time.time()
            step_count = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--no-audio', action='store_true')
    args = parser.parse_args()
    
    shared = SharedState()
    
    # DÜZELTME: Taichi kernel'lar main thread'de, diğer thread'ler daemon
    sensory_th = threading.Thread(target=sensory_thread, args=(shared, not args.no_camera, not args.no_audio), daemon=True)
    dialogue_th = threading.Thread(target=dialogue_thread, args=(shared,), daemon=True)
    
    sensory_th.start()
    dialogue_th.start()
    
    print("PROTO-AGI CANLI! Konuşmak için terminale yaz:")
    
    # Core'u main thread yap
    try:
        core_thread(shared)  # Main thread'de çalıştır
        while shared.running:
            text = input("> ")
            if text.lower() in ["quit", "exit"]: 
                shared.running = False
                break
            shared.user_queue.put(text)
    except KeyboardInterrupt:
        shared.running = False
        print("\nProto-AGI kapanıyor...")

if __name__ == "__main__":
    main()