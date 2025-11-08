#!/usr/bin/env python3
"""
Proto-AGI Taichi Edition – SNN Dil Motoru v2.0 (Tokenizer Entegrasyonu)
Hugging Face tokenizer ile güçlendirilmiş, LLM'siz, SNN tabanlı konuşan AGI.
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
import logging
from transformers import AutoTokenizer
from psinet.core.taichi_neuron import BionicNeuron
from psinet.core.taichi_synapse import BionicSynapse
from psinet.io.realtime_encoders import RealtimeEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FAZ 1: TOKENIZER ALTYAPISI ---
TOKENIZER_MODEL = "dbmdz/bert-base-turkish-cased"
logger.info(f"Hugging Face tokenizer yükleniyor: {TOKENIZER_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
VOCAB_SIZE = tokenizer.vocab_size
logger.info(f"Tokenizer başarıyla yüklendi. Kelime/Token sayısı: {VOCAB_SIZE}")

# Sabitler
MOTOR_NEURONS_COUNT = VOCAB_SIZE # Her token bir motor nörona karşılık gelir
ASSOCIATION_NEURONS_COUNT = 8000
SENSORY_NEURONS_COUNT = 12000
LANGUAGE_NEURONS_COUNT = VOCAB_SIZE # Her token bir dil girdi nöronuna karşılık gelir

class SharedState:
    def __init__(self):
        self.running = True
        self.sensory_queue = queue.Queue(maxsize=10)
        self.spoken_tokens_queue = queue.Queue(maxsize=50) # Artık token ID'leri taşıyor
        self.self_talk_queue = queue.Queue(maxsize=50) # Geri besleme için token ID'leri
        self.lock = threading.Lock()
        self.dopamine = 1.0

def core_thread(shared: SharedState):
    logger.info("Core thread: SNN Dil Motoru v2.0 aktif ediliyor...")
    
    # --- FAZ 2: SNN ENTEGRASYONU ---
    # Nöron Katmanları
    sensory_neurons = BionicNeuron(SENSORY_NEURONS_COUNT, dt=0.001)
    association_neurons = BionicNeuron(ASSOCIATION_NEURONS_COUNT, dt=0.001)
    motor_neurons = BionicNeuron(MOTOR_NEURONS_COUNT, dt=0.001)
    language_neurons = BionicNeuron(LANGUAGE_NEURONS_COUNT, dt=0.001)

    # Sinapslar
    syn_sensory_to_assoc = BionicSynapse(sensory_neurons, association_neurons)
    syn_assoc_to_motor = BionicSynapse(association_neurons, motor_neurons)
    syn_motor_to_assoc = BionicSynapse(motor_neurons, association_neurons, a_post=-0.015)
    syn_lang_to_assoc = BionicSynapse(language_neurons, association_neurons)

    # Decoder Kernel için Taichi alanları
    motor_spikes_field = ti.field(ti.f32, shape=MOTOR_NEURONS_COUNT)
    last_spoken_tokens = -np.ones(10, dtype=np.int32)
    
    @ti.kernel
    def decode_and_speak(dopamine_level: ti.f32):
        # En çok ateşlenen motor nöronunu bul (bu nöronun indeksi = token ID)
        best_token_id = -1
        max_spikes = 0.0
        for i in range(MOTOR_NEURONS_COUNT):
            is_recent = False
            for k in ti.static(range(last_spoken_tokens.shape[0])):
                if i == last_spoken_tokens[k]:
                    is_recent = True
            
            # Dopamin, konuşma isteğini artırır
            current_spikes = motor_spikes_field[i] * dopamine_level
            if not is_recent and current_spikes > max_spikes:
                max_spikes = current_spikes
                best_token_id = i
        
        if max_spikes > 0.8: # Ateşleme eşiği
            shared.spoken_tokens_queue.put(best_token_id)

    while shared.running:
        # Girdileri işle (Dil ve Sensör)
        try:
            token_ids = shared.self_talk_queue.get_nowait()
            lang_rates = np.zeros(LANGUAGE_NEURONS_COUNT, dtype=np.float32)
            lang_rates[token_ids] = 150.0
            language_neurons.apply_poisson_input(lang_rates)
        except queue.Empty:
            language_neurons.update()

        try:
            rates = shared.sensory_queue.get_nowait()
            sensory_neurons.apply_poisson_input(rates)
        except queue.Empty:
            sensory_neurons.update()

        # Simülasyon adımı
        sensory_spikes = sensory_neurons.get_spikes()
        lang_spikes = language_neurons.get_spikes()
        assoc_spikes = association_neurons.get_spikes()
        motor_spikes = motor_neurons.get_spikes()

        psc_to_assoc = syn_sensory_to_assoc.update(sensory_spikes, assoc_spikes) + \
                       syn_motor_to_assoc.update(motor_spikes, assoc_spikes) + \
                       syn_lang_to_assoc.update(lang_spikes, assoc_spikes)
        
        psc_to_motor = syn_assoc_to_motor.update(assoc_spikes, motor_spikes)

        association_neurons.update(psc_to_assoc)
        motor_neurons.update(psc_to_motor)

        motor_spikes_field.from_numpy(motor_neurons.get_spikes().astype(np.float32))

        with shared.lock:
            current_dopamine = shared.dopamine
        
        decode_and_speak(current_dopamine)

        try:
            spoken_id = shared.spoken_tokens_queue.get_nowait()
            np.roll(last_spoken_tokens, 1)
            last_spoken_tokens[0] = spoken_id
        except queue.Empty:
            pass

def sensory_thread(shared: SharedState):
    logger.info("Sensory thread başlatılıyor (simülasyon modu)")
    while shared.running:
        try:
            rates = np.random.rand(SENSORY_NEURONS_COUNT) * 5.0
            shared.sensory_queue.put(rates, timeout=0.1)
            time.sleep(0.05)
        except queue.Full:
            continue

def main():
    shared = SharedState()
    
    threads = [
        threading.Thread(target=sensory_thread, name="Sensory", args=(shared,)),
        threading.Thread(target=core_thread, name="Core", args=(shared,))
    ]
    
    for t in threads:
        t.start()

    # --- FAZ 3: CÜMLE OLUŞTURMA ---
    logger.info("AGI Dil Motoru v2.0 aktif. Cümle üretimi bekleniyor...")
    current_token_ids = []
    last_token_time = time.time()
    period_token_id = tokenizer.convert_tokens_to_ids('.')

    while shared.running:
        try:
            # Kullanıcı girdisini al ve token'lara çevir
            text = input("> ")
            if text.lower() in ["quit", "exit", "q"]:
                break
            
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            shared.self_talk_queue.put(token_ids) # Girdiyi SNN'e besle

            # Dopamini ayarla
            reward = 0.0
            if "iyi" in text.lower() or "doğru" in text.lower(): reward = 1.0
            elif "kötü" in text.lower() or "yanlış" in text.lower(): reward = -1.0
            with shared.lock:
                shared.dopamine = np.clip(1.0 + reward * 0.5, 0.0, 2.0)

        except (EOFError, KeyboardInterrupt):
            break

        # AGI'nin ürettiği token'ları topla ve cümleye çevir
        try:
            while not shared.spoken_tokens_queue.empty():
                token_id = shared.spoken_tokens_queue.get_nowait()
                current_token_ids.append(token_id)
                last_token_time = time.time()
                
                # Kendi kendine konuşma için geri besle
                shared.self_talk_queue.put([token_id])

                # Cümle sonu ise veya zaman aşımı olduysa cümleyi yazdır
                if token_id == period_token_id:
                    sentence = tokenizer.decode(current_token_ids, skip_special_tokens=True)
                    print(f"\nAGI: {sentence}")
                    current_token_ids = []
        except queue.Empty:
            pass
        
        if current_token_ids and time.time() - last_token_time > 1.5:
            sentence = tokenizer.decode(current_token_ids, skip_special_tokens=True)
            print(f"\nAGI: {sentence}")
            current_token_ids = []

    print("\nKapanma sinyali alındı. Proto-AGI durduruluyor...")
    shared.running = False
    for t in threads:
        t.join(timeout=2.0)

    print("Proto-AGI başarıyla kapatıldı.")

if __name__ == "__main__":
    main()